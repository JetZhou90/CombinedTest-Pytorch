import requests,re
from bs4 import BeautifulSoup
import pdb, os
from tqdm import tqdm
import time


def ensure_dir(director):
    if not os.path.exists(director):
        os.makedirs(director)

# url = "http://www.baidu.com"
# url = "http://zxgk.court.gov.cn/zhzxgk/"
# url = "http://zxgk.court.gov.cn/zhzxgk/captcha.do?captchaId=9876fedeb1e14be699810a67e469cde4&random=0.3675069892982034"

url = " https://examapi.sac.net.cn/pages/login/exam_query.html"
times = 2000
save_folder = "result_sac"
ensure_dir(save_folder)

headers = {
    'Connection': 'Keep-Alive',
    'Accept': 'text/html, application/xhtml+xml, */*',
    'Accept-Language': 'en-US,en;q=0.8,zh-Hans-CN;q=0.5,zh-Hans;q=0.3',
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko'
    }

def task(times_i=-1):
    img_name = './%s/%d_jet.png' %(save_folder, times_i)
    if os.path.exists(img_name):
        return "Pass"
    ssion = requests.session()
    # html = ssion.get(url, headers=headers)
    # html.encoding='utf-8'
    # soup = BeautifulSoup(html.text, 'html.parser')
    # all_imgs = soup.find_all("img")


    # if len(all_imgs):
    # if times_i != -1:
        # img_url = all_imgs[-1]["src"]
    img_url = ""
    t = ssion.get(url+img_url, stream=True, headers=headers)
    if len(t.content) > 1e4 or len(t.content)<4e3:
        print("Fail")
        return "Fail"
    if times_i != -1:
        with open(img_name, 'wb') as f:
            for chunk in t.iter_content(chunk_size=128):
                f.write(chunk)
    # return "Done"
    # else:
    #     print("failed")
    #     return "Fail"
while True:
    for times_i in tqdm(range(times)):
        flag = task(times_i=times_i)
        if flag == "Fail":
            break
        elif flag == "Done":
            time.sleep(1)
        # pdb.set_trace()
        # t.encoding='utf-8'
        # soup=BeautifulSoup(t.text,'html.parser')
        # image_name = url.split('/')[-1]
        # with open('./img/%s' % image_name, 'wb') as f:
        # for chunk in r.iter_content(chunk_size=128):
        #     f.write(chunk)
        #     print('Saved %s' % image_name)
    time.sleep(10)
    # pdb.set_trace()
