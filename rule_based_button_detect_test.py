import numpy as np
from skimage import morphology
import skimage.measure
import cv2
import shutil
import sys
sys.path.append(r"..\common\py_util")
from util import *
output_folder = ensure_dir("rule_based_button_detect")


def button_detect(img_path, write_file=True):
    kernelSize = 31
    # 由于可能存在图片名冲突的问题，将结果文件夹的名字命名为: 图片集名-图片名 例:DocBank_samplesColor-2.tar_1801.00617.gz_idempotents_arxiv_4
    imgPath, img_name = os.path.split(os.path.split(img_path)[0])
    img_output_folder = ensure_dir(os.path.join(
        output_folder, img_name))
    shutil.rmtree(img_output_folder)

    def write_img(img, name):
        cv2.imwrite(ensure_parent(os.path.join(
            img_output_folder, name)), img)

    def cv_close(img, kernel_size):
        kernel = np.ones((kernel_size), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    def line_contour(img, cnt_type=cv2.RETR_EXTERNAL, hor_kernal_size=(13, 1), vet_kernal_size=(1, 11), debug_level=2, name="", contours=True):
        if contours:
            contours = cv2.findContours(
                img, cnt_type, cv2.CHAIN_APPROX_SIMPLE)[0]
            img_contours_mask = np.zeros_like(img)
            cv2.drawContours(img_contours_mask, contours, -1, (255), 1)
        else:
            img_contours_mask = img

        det_hor_cnt, det_vet_cnt = line_det_hor_vet(
            img_contours_mask, hor_kernal_size=hor_kernal_size, vet_kernal_size=vet_kernal_size)

        # det_hor_cnt = np.minimum(det_hor, img_contours_mask)
        # det_vet_cnt = np.minimum(det_vet, img_contours_mask)

        lines_cnt = np.maximum(det_hor_cnt, det_vet_cnt)

        if name != "":
            if debug_level >= 1:
                write_img(lines_cnt, f"_bw_det_lines_cnt_{name}.png")
            if debug_level >= 2:
                # write_img(det_hor, f"_bw_det_hor_{name}.png")
                # write_img(det_vet, f"_bw_det_vet_{name}.png")
                write_img(img_contours_mask, f"img_contours_mask_{name}.png")
                write_img(det_hor_cnt, f"_bw_det_hor_cnt_{name}.png")
                write_img(det_vet_cnt, f"_bw_det_vet_cnt_{name}.png")

        return lines_cnt

    def rm_kernal(slim_kernal, im_bin, fat_kernal=None, name="", **kwargs):
        detected_slim_lines = cv2.morphologyEx(
            im_bin, cv2.MORPH_OPEN, slim_kernal, iterations=2)
        used_lines = detected_slim_lines
        if fat_kernal is not None:
            detected_fat_lines = cv2.morphologyEx(
                im_bin, cv2.MORPH_OPEN, fat_kernal, iterations=2)
            used_lines = detected_slim_lines - detected_fat_lines
        im_bin_rm = im_bin - used_lines
        # im_bin_rm = 255 - im_bin_rm

        if name != "":
            write_img(im_bin, name + "_bw_rev.png")
            write_img(detected_slim_lines, name + "_detected_slim_lines.png")
            if fat_kernal is not None:
                write_img(detected_fat_lines, name + "_detected_fat_lines.png")
                write_img(used_lines, name + "_used_lines.png")

        return used_lines, im_bin_rm

    def rm_kernal_size(img, kernal_size):
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernal_size)
        return rm_kernal(hor_kernel, img)

    def line_det_hor_vet(img, hor_kernal_size=(13, 1), vet_kernal_size=(1, 11), name=""):
        # hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, hor_kernal_size)
        # det_hor, img_rmhor = rm_kernal(
        #     hor_kernel, img)
        det_hor, img_rmhor = rm_kernal_size(img, hor_kernal_size)

        # vet_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, vet_kernal_size)
        # det_vet, im_bin_rmvet = rm_kernal(
        #     vet_kernel, img)

        det_vet, im_bin_rmvet = rm_kernal_size(img, vet_kernal_size)

        return det_hor, det_vet

    def conv_thres(im_bin, kernal, thres, op, name="", **kwargs):
        im_conv = cv2.filter2D(im_bin, -1, kernal)
        t, im_conv_thres = cv2.threshold(
            im_conv, thres, 255, cv2.THRESH_BINARY)
        if name != "":
            write_img(im_conv_thres,  f"{name}.png")
        im_conv_thres = op(im_conv_thres, im_bin, **kwargs)
        return im_conv_thres

    def mean_pool_thres(im_bin, kernal_shape, thres, op, name=""):
        kernal = np.ones(kernal_shape, np.float)
        kernal /= kernal.sum()
        return conv_thres(im_bin, kernal, thres, op, name)

    # alone point
    def rm_alone(im_bin, name=""):
        return mean_pool_thres(im_bin, (3, 3), thres=255 * 8 / 16 + 1, op=np.minimum, name=name + f"_im_rm_alone")

    def bbox_connected_connectedComponentsWithStats(im_contour, thickness=1, name=""):
        num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(
            im_contour, connectivity=8, ltype=cv2.CV_32S)
        bboxes = [i[0:4] for i in stats]
        im_bbox = img_color.copy()
        # write_img(img_color, name + f"origin.png")
        bboxes_prune = []
        for bbox in bboxes:
            (x, y, w, h) = bbox
            if w < 3 or h < 3:
                continue
            cv2.rectangle(im_bbox, (x, y), (x+w, y+h),
                          (255, 0, 0), thickness=thickness)
            bboxes_prune.append([x, y, x+w, y+h])
        write_img(im_bbox, name + f"_im_bbox.png")
        return bboxes_prune

    def gradient_thres(im_bin, kernal, thres, name="", **kwargs):
        im_bin_float = im_bin.astype(np.float)
        im_conv = cv2.filter2D(im_bin_float, -1, kernal, **kwargs)
        im_conv = im_conv.astype(np.uint8)
        t, im_conv_thres = cv2.threshold(
            im_conv, thres, 255, cv2.THRESH_BINARY)
        if name != "":
            write_img(im_conv_thres,  f"{name}.png")
        return im_conv_thres

    def line_detect_filter(img, kernelSize, debug_level=1, name=""):
        binary_line = np.zeros_like(img)
        anchor = [0, 0]
        hor_kernel = (kernelSize, 1)
        vet_kernel = (1, kernelSize)
        thresh = 1
        # 检测竖线
        long_lines_vet_candidate = 255 - \
            gradient_thres(img, np.array(
                [[1], [-1]]), thresh, anchor=tuple(anchor))

        long_lines_vet, rm_lines_vet = rm_kernal_size(
            long_lines_vet_candidate, vet_kernel)
        write_img(long_lines_vet_candidate, "long_lines_vet_candidate.png")
        write_img(long_lines_vet, "long_lines_vet.png")

        # 检测横线
        long_lines_hor_candidate = 255 - \
            gradient_thres(img, np.array(
                [[1, -1]]), thresh, anchor=tuple(anchor))
        long_lines_hor, rm_lines_hor = rm_kernal_size(
            long_lines_hor_candidate, hor_kernel)
        write_img(long_lines_hor_candidate, "long_lines_hor_candidate.png")
        write_img(long_lines_hor, "long_lines_hor.png")

        long_lines = np.maximum(long_lines_vet, long_lines_hor)
        write_img(long_lines, "long_lines.png")
        hist = cv2.calcHist([img], [0], None, [256], [0.0, 255.0])
        countAll = img.shape[0]*img.shape[1]
        countthreshold = 0.1
        countthresholdSingle = 0.05
        countRange = []
        countSum = 0
        countStart = 0
        for hist_i in range(len(hist)):
            countSum += hist[hist_i]
            countEnd = hist_i
            if countSum > countAll*countthreshold:
                if hist[hist_i] > countAll*countthresholdSingle:
                    countEnd = hist_i-1
                    if countEnd < 0:
                        countRange.append([0, 0])
                    else:
                        countRange.append([countStart, countEnd])
                        countRange.append([hist_i, hist_i])
                else:
                    countRange.append([countStart, countEnd])
                countStart = hist_i+1
                countSum = 0
            if hist_i == 255:
                countRange.append([countStart, 255])

        for i in countRange:
            img_in_range = cv2.inRange(img, i[0], i[1])
            img_in_range = np.minimum(img_in_range, long_lines)

            # rm background
            img_in_range_bg = np.zeros_like(img_in_range)
            if (img_in_range == 255).sum() > 50 * 50:
                num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(
                    img_in_range, connectivity=4, ltype=cv2.CV_32S)
                for j in range(1, num_labels):
                    x, y, w, h, area = stats[j]
                    if area > 0.2 * w * h and area > 50 * 50 and w >= 5 and h >= 5:
                        img_in_range_bg[labels == j] = 255

            img_in_range_front = img_in_range - img_in_range_bg
            det_line = img_in_range_front
            binary_line = np.maximum(binary_line, det_line)
            if name != "":
                if debug_level >= 2:
                    write_img(img_in_range_bg,
                              f"{name}_img_in_range_bg{i+1}.png")
                    write_img(img_in_range_front,
                              f"{name}_img_in_range_front{i+1}.png")
                    write_img(det_line, f"{name}_det_line{i+1}.png")
                    write_img(img_in_range, f"{name}_img_in_range{i+1}.png")

        return binary_line

    img_color = cv2.imread(img_path)
    write_img(img_color, "_orig.png")

    im_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    write_img(im_gray, "im_gray.png")

    binary_line = line_detect_filter(
        im_gray, kernelSize, debug_level=0, name="gray")
    write_img(binary_line, "binary_line.png")

    out_contour_line = binary_line
    im_bin_adpt = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 3, 4)

    write_img(im_bin_adpt, "_binary.png")
    # im_bin_adpt_rev = 255 - im_bin_adpt
    im_bin = im_bin_adpt

    im_bin_adpt_rev = 255 - im_bin_adpt
    det_hor, det_vet = line_det_hor_vet(
        im_bin_adpt_rev, hor_kernal_size=(kernelSize, 1), vet_kernal_size=(1, kernelSize))
    det_line = np.maximum(det_hor, det_vet)
    write_img(det_line, "det_line.png")

    det_line = np.maximum(det_line, out_contour_line)
    write_img(det_line, "combine_det_line.png")

    det_line_rev = 255 - det_line
    # im_bin_rm_line = im_bin + det_line
    im_bin_rm_line = np.maximum(im_bin, det_line)

    write_img(im_bin_rm_line, "_im_bin_rm_line.png")
    im_bin = im_bin_rm_line

    kernel = np.ones((3, 3), np.uint8)
    im_grad = cv2.morphologyEx(im_bin, cv2.MORPH_GRADIENT, kernel)
    write_img(im_grad, "_gradient.png")
    # im_grad = 255 - im_bin

    im_bin_rm = im_grad
    # im_bin_rm = iter_rm_line_contour(
    #     im_grad, cnt_type=cv2.RETR_LIST, iterations=20)
    # write_img(im_bin_rm, "_bw_rmline.png")

    im_rm_alone = rm_alone(im_bin_rm)
    write_img(im_rm_alone, "im_rm_alone.png")
    im_grad = im_rm_alone
    im_grad_close = im_grad

    write_img(im_grad_close, f"im_grad_close.png")
    roi_bboxes = bbox_connected_connectedComponentsWithStats(im_grad_close)

    # For testing button classification
    # It is commented as no writting the text_detect_cache.txt
    # -----------------------------------------------------------------
    if write_file:
        bboxes2str = ["\t".join(['text', '1.0', str(bbox[0]), str(
            bbox[1]), str(bbox[2]), str(bbox[3])]) for bbox in roi_bboxes]
        write_lines(os.path.join(img_output_folder,
                                 "text_detect_cache.txt"), bboxes2str)
    # ------------------------------------------------------------------
    with open(os.path.join(output_folder, "label_files.list"), "a", encoding='utf-8') as f:
        f.write(os.path.join(os.getcwd(),img_output_folder,"text_detect_cache.txt") + "\n")

    return img_color, roi_bboxes

def get_img_path(file_list):
    use_img_paths = []
    lable_files = []
    for labelPath in tqdm.tqdm(file_list):
        imgRelativePath, imgName = os.path.split(
            read_list(labelPath)[0])  # label文件内的第一行为图片相对于数据库/datas的相对路径
        folderRelativePath, folderName = os.path.split(imgRelativePath)
        imgAbsolutePath = os.path.join(
            labelPath[:labelPath.find(imgRelativePath)]+imgRelativePath, imgName)
        use_img_paths.append(imgAbsolutePath)
    return use_img_paths


if __name__ == "__main__":
    # dataset_path = r"D:\Research\textDetect\DocBank\DocBank_samples\datas\textDetect\detect\dataset.list"
    dataset_path = r"D:\Research\datas\iconAndTextDetect\detect\commonUseInterface\label.list"
    dataset_list = read_list(dataset_path)
    use_img_paths = []
    lable_files = []
    if dataset_list[0][-4:] == 'list':
        for dataset_path in tqdm.tqdm(dataset_list):
            labelList = read_list(dataset_path, func=lambda l: l.split()[
                0], errors="ignore")
            use_img_paths.extend(get_img_path(labelList))
    else:
        use_img_paths= get_img_path(dataset_list)

    write_lines(os.path.join(output_folder, "img_path.list"), use_img_paths)
    open(os.path.join(output_folder, "label_files.list"), "w", encoding='utf-8')
    # for path_i in use_img_paths:
    #     button_detect(path_i, True)
    # r = list(tqdm.tqdm(map(text_detect, use_img_paths)))
    with Pool(processes=cpu_count()) as pool:
        r = list(tqdm.tqdm(pool.imap(button_detect, use_img_paths)))
        pool.close()
        pool.join()
