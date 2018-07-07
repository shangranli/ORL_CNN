from urllib import request
import zipfile


DATA_URL = r'https://nj02cm01.baidupcs.com/file/ce162d57e03568f6f0d0fb93d9301182?bkt=p3-000030cb5b9b83da1e734410ca5a597cc058&fid=1872757597-250528-896148336640813&time=1530948955&sign=FDTAXGERLQBHSK-DCb740ccc5511e5e8fedcff06b081203-b21QtqwvdE3gCVrroeGf3K0IMsg%3D&to=88&size=4234685&sta_dx=4234685&sta_cs=3&sta_ft=zip&sta_ct=0&sta_mt=0&fm2=MH%2CYangquan%2CAnywhere%2C%2Chebei%2Ccmnet&vuk=1872757597&iv=0&newver=1&newfm=1&secfm=1&flow_ver=3&pkey=000030cb5b9b83da1e734410ca5a597cc058&sl=76480590&expires=8h&rt=sh&r=159346909&mlogid=4352131411276400783&vbdid=3769895679&fin=ORL.zip&fn=ORL.zip&rtype=1&dp-logid=4352131411276400783&dp-callid=0.1.1&hps=1&tsl=80&csl=80&csign=I7WmcEoKFY56sDqRiSu32AyaQaM%3D&so=0&ut=6&uter=1&serv=0&uc=3572096006&ic=3940846637&ti=26fa64dbec2882241575d857e0294f7c950f6948225381ad305a5e1275657320&by=themis'

zip_save_path=r'C:\Users\Administrator\Desktop\orl.zip'
unzip_path=r'C:\Users\Administrator\Desktop'

flag_num = 0

def download():

    def progress(block_num, block_size, total_size):
        global flag_num
        percent = float(block_num* block_size) / float(total_size) * 100.0
        flag_num+=1
        if flag_num %10 == 0 or percent==1:
            print("downloaded:%.2f %%"%percent)
    
    filepath, _ = request.urlretrieve(DATA_URL,zip_save_path,progress)

    orl_zip = zipfile.ZipFile(zip_save_path)
    orl_zip.extractall(unzip_path)

    data_path=unzip_path + r'\ORL'

    return data_path

if __name__ == '__main__':
    download()
