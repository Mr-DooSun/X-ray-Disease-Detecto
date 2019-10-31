import pydicom as dicom
from matplotlib import pyplot as plt
from tkinter import filedialog
from tkinter import *
import cv2

if __name__=="__main__":

	sDCM = Tk()
	#디렉토리 다얄로그 오픈 

	sDCM.filename = filedialog.askopenfilename(initialdir = "/",title="Select file", filetypes = (("dcm files","*.dcm"),))
	#print(sDCM.filename)
	
	d = dicom.read_file(sDCM.filename)
	# Mate=plt.imsave(sDCM.filename[0:len(sDCM.filename) - 4] + ".png",d.pixel_array,cmap=plt.cm.bone)
	# Mate=cv2.imread(sDCM.filename[0:len(sDCM.filename) - 4] + ".png",cv2.IMREAD_COLOR)
	print(d)
	print("patient's sex : " +d[0x0010,0x0040].value)
	print("patient's age : " +d[0x0010,0x1010].value)
	print("body part : " +d[0x0018,0x0015].value)
	print("image size : %d x %d" %(d[0x0028,0x0010].value, d[0x0028,0x0011].value))
	print("lossy image com : " +d[0x0028,0x2114].value)
	print("pixel data size : %d bytes" %(len(d[0x7fe0,0x0010].value)))

	cv2.imshow("image",Mate)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	# dcm to png 