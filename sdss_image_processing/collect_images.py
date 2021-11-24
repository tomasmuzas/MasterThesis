import requests
import skimage.io
from io import BytesIO
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
import os

def getJpegImgCutout(ra, dec, scale=0.1, width=424, height=424, opt="", query="", dataRelease=None):
    
    url = "http://skyservice.pha.jhu.edu/DR7/ImgCutout/getjpeg.aspx?"

    url = url + 'ra=' + str(ra) + '&'
    url = url + 'dec=' + str(dec) + '&'
    url = url + 'scale=' + str(scale) + '&'
    url = url + 'width=' + str(width) + '&'
    url = url + 'height=' + str(height) + '&'
    url = url + 'opt=' + opt + '&'
    # url = url + 'query=' + query + '&'
    # url = url + "TaskName=Compute.SciScript-Python.SkyServer.getJpegImgCutout&"
    
    acceptHeader = "text/plain"
    headers = {'Content-Type': 'application/json', 'Accept': acceptHeader}
    
    response = requests.get(url,headers=headers, stream=True)
    if response.status_code != 200:
        if response.status_code == 404 or response.status_code == 500:
            print("Error when getting an image cutout.\nHttp Response from SkyServer API returned status code " + str(response.status_code) + ". " + response.reason);
            raise Exception("Error when getting an image cutout.\nHttp Response from SkyServer API returned status code " + str(response.status_code) + ". " + response.reason);
        else:
            print("Error when getting an image cutout.\nHttp Response from SkyServer API returned status code " + str(response.status_code) + ":\n" + response.content.decode());
            raise Exception("Error when getting an image cutout.\nHttp Response from SkyServer API returned status code " + str(response.status_code) + ":\n" + response.content.decode());

    return skimage.io.imread( BytesIO( response.content))

ra_dec_data = pd.read_csv("../../data/AllGalaxiesRaDec.csv")
failed_galaxies_data = ra_dec_data

def save_image(i):
    if i % 100 == 0:
        print(i)
    try:
        if(os.path.exists("../../data/galaxy_images/{0}.jpg".format(failed_galaxies_data["Id"][i]))):
            return
        image = getJpegImgCutout(failed_galaxies_data["ra"][i], failed_galaxies_data["dec"][i], scale=failed_galaxies_data["petroR90_r"][i] * 0.024)
        skimage.io.imsave("../../data/galaxy_images/{0}.jpg".format(failed_galaxies_data["Id"][i]), image)
        print("Downloaded galaxy {0}".format(failed_galaxies_data["Id"][i]))
    except Exception as e:
        print(e)
        print("failed to get image for galaxy {0}".format(failed_galaxies_data["Id"][i]))

# num_cores = multiprocessing.cpu_count()

# results = Parallel(n_jobs=num_cores)(delayed(save_image)(i) for i in range(0, len(failed_galaxies_data)))

for i, galaxy in failed_galaxies_data.iterrows():
    save_image(i)