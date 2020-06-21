import os
from multiprocessing import Process

PATH = '/scidatasm/michalf'

def download(url, name):
    os.system(f'wget -O {PATH}/{name} {url}')
    
def move(p1, p2):
    os.system(f'mv {p1} {p2}')
    
def untar(url, name):
    os.system(f'tar -C {PATH} -xvzf {PATH}/{name}')
    os.system(f'rm -rf {PATH}/{name}')
    
def creat_name(i):
    return str(i) + '.tar.gz'

def run(proc):
    for p in proc:
        p.start()
    for p in proc:
        p.join()

if __name__ == '__main__':
    urls = ["'https://download.xview2.org/train_images_labels_targets.tar.gz?Expires=1590533296&Signature=HFjgmyGtsszhhpZ3-nwgUW~oekJgXrmRW3CH1LxXsA8Wie9~MQ6IprPtCi2SEX3x~ov2cmn4WgSpgllf2g9kE3NCBG-wDa46Wa-3kYa4o9FmVAFWnJ6iaZaQ0ANpgFLGLCm4ZwqiYLyBf~zoUgVL4PIVxDzJpD4~Y~P7doT9bIV9AFldEAPgGcKpUf9SNvvpj8-jpLMrQbxbnytc9WfIWehVDeISMtk0m~n158DusPTbDks9bXWQkFjo3QTSRB6ngZPaAAk9yt7xQ8TNN5HYhHs6DnhIE~svz2iOJZRtf6MApt7aaW8biS6qFH0V4y6LV-FUyNpaHlhGoovfnGC8XA__&Key-Pair-Id=APKAIKGDJB5C3XUL2DXQ'",
"'https://download.xview2.org/tier3.tar.gz?Expires=1590533296&Signature=Sr-G5on3i8sCn~dpSXkc9X8ivRZ6QbUiAbAaac20NMkN4TZNw-5NcuBKfBdp6bbyH0XGA72DCNzbLqGWW9dfKuCA3mpp2LP-R7x9o1W7hBfXGPZ8Oh9YZGRUlXB~mMCIi4L3CZ7o~KYsgDnNI6Pm8cFDUJf1ltCpY975XG-eCk96DjMtJTGMmPdkIng-oWkw16bCc9InIvuRruAtSQw4BFR0V4xHc6ghJatFWOIP7EFhwUdFbj76MQBdh-CeqQFkS6XFmUs~PV4Bc0L8BSMZA0vzoSHPGOgunbMuAVDXsCGYC3MbyGZQI2m8QsWl7NH82UxZ0ZF8LJp-4thO3mZHRw__&Key-Pair-Id=APKAIKGDJB5C3XUL2DXQ'"]
        
    # Download
    proc = [Process(target=download, args=(url, creat_name(i),)) for (i, url) in enumerate(urls)]
    run(proc)
    
    # Untar
    proc = [Process(target=untar, args=(url, creat_name(i),)) for (i, url) in enumerate(urls)]
    run(proc)
      
    # Move train to tier3
    proc = [Process(target=move, args=(f'{PATH}/train/images/*', f'{PATH}/tier3/images/',)),
            Process(target=move, args=(f'{PATH}/train/labels/*', f'{PATH}/tier3/labels/',)),]
    run(proc)
    
    os.system(f'rm -rf {PATH}/train')
    os.system(f'mv {PATH}/tier3 {PATH}/train')
    os.system('python create_mask.py')
    