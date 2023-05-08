import os
from IPython.display import display
from matplotlib import pyplot as plt 
import matplotlib.gridspec as gridspec
import random
from pylab import imread
from skimage.transform import rescale, resize
import cv2 
import skimage
from skimage import measure
from skimage.measure import label, regionprops
from skimage.segmentation import mark_boundaries
from skimage.morphology import (erosion, dilation, opening, closing, white_tophat)
from skimage.morphology import black_tophat, skeletonize, convex_hull_image  
from skimage.morphology import disk
from scipy import ndimage
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
class HelperFunction:
  @classmethod
  def ConnectToDataFolder(cls,FolderPath):
    """
      Connect to the Data Folder

      Parameters:
          FolderPath (str): The path to the Data Folder

      Returns:
          None
    """
    print("=" * 80, "- Begin: ConnectToDataFolder")

    if (os.path.isdir(FolderPath)):
      os.chdir(FolderPath)
      print("List of Data in the Folder :")
      display(os.listdir())
    else:
      print(f"{FolderPath} is not correct, please check the folder again")
    
    print("=" * 80, "- Finish: ConnectToDataFolder")

  @classmethod
  def GetSubFiles(cls,dir, ExtensionList = []):
      print("=" * 80, "- Begin: GetSubFiles")

      "Get a list of immediate subfiles"
      all_names = next(os.walk(dir))[2]
      subfile_names = []
      if(len(ExtensionList) > 0):
        for filename in all_names:
          fname, fextension = os.path.splitext(filename)
          if any(ext in fextension for ext in ExtensionList):
            subfile_names.append(dir + "/" + filename)
      else:
        subfile_names = [dir + "/" + x for x in all_names]
      
      print(f"There are {len(subfile_names)} files are found")
      subfile_names.sort()
      print("Here is some samples :")
      [print(x) for x in subfile_names[0: min(len(subfile_names), 5)]]

      print("=" * 80, "- End: GetSubFiles")
      return subfile_names
  @classmethod
  def ShowImage(cls,ImageList, nRows = 1, nCols = 2, ImageTitleList = []):    
      gs = gridspec.GridSpec(nRows, nCols)     
      plt.figure(figsize=(20,20))
      for i in range(len(ImageList)):
          ax = plt.subplot(gs[i])
          ax.set_xticklabels([])
          ax.set_yticklabels([])
          ax.set_aspect('equal')

          plt.subplot(nRows, nCols,i+1)

          image = ImageList[i].copy()
          if (len(image.shape) < 3):
              plt.imshow(image, plt.cm.gray)
          else:
              plt.imshow(image)
          if(len(ImageTitleList)  > 0):
            plt.title("Image " + str(ImageTitleList[i]))
          else:
            plt.title("Image " + str(i))

          plt.axis('off')

      plt.show()

  @classmethod
  def ShowRandomImage(cls,DatasetFiles, nRows = 1, nCols = 5, seedNo = 10):
    """
      Show Random Images from the Dataset Files

      Input:
          DatasetFiles: List of Dataset Files
          nRows: Number of Rows
          nCols: Number of Columns
          seedNo: Seed Number

      Output:
          FileNameList: List of File Names
          ImageList: List of Images
    """
    print("=" * 80, "- Begin: ShowRandomImage")

    nFile = nRows * nCols

    random.seed(seedNo)
    FileNameList = random.sample(DatasetFiles, nFile)
    ImageList = []
    for filepath in FileNameList:
      image_RGB = imread(filepath)
      ImageList.append(image_RGB)
    
    cls.ShowImage(ImageList, nRows, nCols, FileNameList)

    print("=" * 80, "- Finish: ShowRandomImage")
    return FileNameList, ImageList

  @classmethod
  def ResizeImage(cls,image, wresize = 0, hresize = 0):
    OrigWidth, OrigHeight = float(image.shape[1]), float(image.shape[0])

    if((wresize == 0) & (hresize == 0)):
        return image
    if(wresize == 0):
        wresize = int((OrigWidth * hresize)/OrigHeight)
    if(hresize == 0):
        hresize = int((OrigHeight * wresize)/OrigWidth)
    resize_image = cv2.resize(image, (wresize, hresize), interpolation = cv2.INTER_NEAREST) 
    return resize_image

  @classmethod
  def ConvertColorSpaces(cls,image, ColorSpace = "GRAY", display = 1):
    """
      Convert Color Spaces

      Parameters:
          image (numpy array): The image
          ColorSpace (str): The Color Space
          display (int): Display the image or not

      Returns:
          image_convert (numpy array): The converted image
    """

    ImageTitleList = ["RGB", ColorSpace]

    if(ColorSpace.upper() == "HSV"):
      image_convert = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
      ChannelList = ["Chrominance Hue" , "Chrominance Saturation", "Luminance Value"]
    elif(ColorSpace.upper() == "YCRCB"):
      image_convert = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
      ChannelList = ["Chrominance Red" , "Chrominance Blue", "Luminance Y"]
    elif(ColorSpace.upper() == "LAB"):
      image_convert = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
      ChannelList = ["Chrominance a(Green To Red)" , "Chrominance b(Blue To Yellow)", "Luminance L"]
    else:
      image_convert = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if(display):
      if(len(image_convert.shape) == 3):  
        cls.ShowImage([image, image_convert, image_convert[:,:,0], image_convert[:,:,1], image_convert[:,:,2]], 1, 5, 
                  ImageTitleList + ChannelList)
      else:
        cls.ShowImage([image, image_convert], 1, 5, ImageTitleList)

    return image_convert

  @classmethod
  def ShowHistogram(cls,image, Title = "Color Histogram", ChannelList = ["Channel 1", "Channel 2", "Channel 3"]):
    
    """
      Plot the histogram of the image

      Parameters
      ----------
      image : numpy array
          The image to be plot
      Title : string
          The title of the plot
      ChannelList : list of string    
          The list of channel names

      Returns
      -------
      None
    """
    fig, ax = plt.subplots(figsize=(20, 10))

    if(len(image.shape) == 3):
      color = ('r', 'g', 'b')
      for channel,col in enumerate(color):
          histr = cv2.calcHist([image],[channel],None,[256],[0,256])
          ax.plot(histr,color = col, label = ChannelList[channel])
          
      plt.title(Title)
      plt.legend()
    else:
      histr = cv2.calcHist([image],[0],None,[256],[0,256])
      plt.plot(histr,color = "gray", label = "Gray")
      plt.title("Gray Histogram")
      plt.legend()

    # Set axis ranges; by default this will put major ticks every 25.
    ax.set_xlim(0, 255)

    # Change major ticks to show every 20.
    ax.xaxis.set_major_locator(MultipleLocator(20))

    # Change minor ticks to show every 5. (20/4 = 5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))

    # Turn grid on for both major and minor ticks and style minor slightly
    # differently.
    ax.grid(which='major', color='#CCCCCC', linestyle='--')
    ax.grid(which='minor', color='#CCCCCC', linestyle=':')
    plt.show()

  @classmethod
  def SegmentByThresh(cls,image, channel, segment_range = [], display = 1):
    """
      Segmentation by thresholding

      Parameters
      ----------
      image : numpy array
          The image to be segmented
      channel : numpy array
          The channel to be segmented
      segment_range : list
          The range of the channel to be segmented
      display : int   
          Display the segmented image or not

      Returns
      -------
      image_mask : numpy array
          The segmented mask
    """
    if(len(segment_range) == 0):
      thresh, image_mask = cv2.threshold(channel,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
      print(f"Otsu Threshold : {thresh}")
    else:
      image_mask = ((channel > segment_range[0]) & (channel < segment_range[1]))

    image_segment = cv2.bitwise_and(image, image, mask = image_mask.astype(np.uint8))

    ImageTitleList = ["Color Image", "Segmented Channel", "Segmented Mask", "Segmented Color Image"]
    if(display):
      cls.ShowImage([image, channel, image_mask, image_segment], 1, 5, ImageTitleList)

    image_mask = image_mask.astype(bool)
    return image_mask
  @classmethod
  def GetLargestBinaryArea(cls,image, image_mask, display = 1):
    """
      Get the largest binary area from the image mask

      Parameters:
          image (numpy array): Image
          image_mask (numpy array): Image Mask
          display (int): Display the image or not

      Returns:
          image_mask (numpy array): Image Mask with largest area

    """

    labels_mask = measure.label(image_mask)                       
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
    labels_mask[labels_mask!=0] = 1
    image_mask = labels_mask
    
    image_segment = cv2.bitwise_and(image, image, mask = image_mask.astype(np.uint8))
    ImageTitleList = ["Color Image", "Segmented Mask", "Segmented Color Image"]
    
    if(display):
      cls.ShowImage([image, image_mask, image_segment], 1, 5, ImageTitleList)

    return image_mask


  @classmethod
  def LabelObjectByMask(cls,image_input, image_mask, type = "BBox", color = (0,255,0)):
      """
      Label the object by the mask

      Parameters:
          image_input (numpy array): Image
          image_mask (numpy array): Image Mask
          type (string): Type of the label including [BBox, Boundary, Fill]
          color (tuple): Color of the label

      Returns:
          image_output (numpy array): Image with label
      """
      image_output = image_input.copy()

      Marker_Size = int(image_input.shape[0] * 0.1)
      Marker_Thick = int(image_input.shape[0] * 0.01)
      BBoxThick = int(image_input.shape[0] * 0.01)
      ContourThick = int(image_input.shape[0] * 0.01)

      label_img = label(image_mask)
      regions = regionprops(label_img)
      for props in regions:
          minr, minc, maxr, maxc = props.bbox
          left_top = (minc, minr)
          right_bottom = (maxc, maxr)
          at_row, at_col = props.centroid

          if(type == "BBox"):
            cv2.rectangle(image_output,left_top, right_bottom, color ,BBoxThick)

          if(type == "Boundary"):
            contours, heirarchy = cv2.findContours(image_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image_output, contours, -1, color, ContourThick)
            cv2.drawMarker(image_output, (int(at_col), int(at_row)),color, markerType=cv2.MARKER_STAR, 
                            markerSize= Marker_Size, thickness= Marker_Thick, line_type=cv2.LINE_AA)
            
          if(type == "Fill"):
            image_output[image_mask > 0] = color
              
      return image_output
  @classmethod
  def IntensityTransformation(cls,image, gamma = 0.3, display = 1):
    """
      Intensity Transformation

      Parameters
      ----------
      image : numpy array
          The image to be transformed
      gamma : float
          The gamma value
      display : int
          Display the transformed image or not

      Returns
      -------
      image_gamma_corrected : numpy array
          The transformed image

    """
    image_gamma_corrected = np.array(255*(image / 255) ** gamma, dtype = 'uint8')
    cls.ShowImage([image, image_gamma_corrected], 1, 5, ["Original Image", f"Gamma = {gamma}"])
    if(display):
      GammaImageList = []
      TitleList = []
    
      for g in [0.3, 0.5, 0.9,  1.2, 2.2]:
        # Apply gamma correction.
        gamma_corrected = np.array(255*(image / 255) ** g, dtype = 'uint8')
        GammaImageList.append(gamma_corrected)
        TitleList.append(f"Gamma = {g}")

      cls.ShowImage(GammaImageList, 1, 5, TitleList)
    return image_gamma_corrected


  # Morphology
  @classmethod 
  def Morphology(cls,image, image_mask, morType = "erosion", size = 3, display = 1):
    
    """
      Morphology

      Parameters
      ----------
      image : numpy array
          The image to be transformed
      image_mask : numpy array
          The image mask
      morType : string
          The type of morphology
      size : int
          The size of the kernel
      display : int
          Display the transformed image or not

      Returns
      -------
      result : numpy array
          The transformed image
      image_mask : numpy array
          The transformed image mask
    """
    image_mask = image_mask.astype(np.uint8)
    kernel = disk(abs(size))

    if(morType == "erosion"):
      result = erosion(image_mask, kernel)
    if(morType == "dilation"):
      result = dilation(image_mask, kernel)
    if(morType == "opening"):
      result = opening(image_mask, kernel)
    if(morType == "closing"):
      result = closing(image_mask, kernel)
    if(morType == "gradient"):
      result = cv2.morphologyEx(image_mask, cv2.MORPH_GRADIENT, kernel)
    if(morType == "tophat"):
      result = white_tophat(image_mask, kernel)
    if(morType == "blackhat"):
      result = black_tophat(image_mask, kernel)
    if(morType == "fillhole"):
      result = ndimage.binary_fill_holes(image_mask).astype(int)
    if(morType == "skeleton"):
      result = skeletonize(image_mask == 1)
    if(morType == "convexhull"):
      result = convex_hull_image(image_mask == 1)
    
    #mask of image
    image_mask_color = cv2.bitwise_and(image, image, mask = result.astype(np.uint8))

    if(display):

      im_erosion = erosion(image_mask, kernel)
      im_dilation = dilation(image_mask, kernel)
      im_opening = opening(image_mask, kernel)
      im_closing = closing(image_mask, kernel)
      im_gradient = cv2.morphologyEx(image_mask, cv2.MORPH_GRADIENT, kernel)
      im_tophat = white_tophat(image_mask, kernel)
      im_blackhat = black_tophat(image_mask, kernel)
      im_fillhole = ndimage.binary_fill_holes(image_mask).astype(int)
      im_skeleton = skeletonize(image_mask == 1)
      im_convexhull = convex_hull_image(image_mask == 1)

      cls.ShowImage([image, result, image_mask_color], 1, 5, ImageTitleList= ["Color Image", "Mask", "Segment By Mask"])
      cls.ShowImage([im_erosion, im_dilation, im_opening, im_closing, im_gradient], 1, 5, 
                ImageTitleList= ["Erosion", "Dilation", "Opening", "Closing", "Gradient"])
      cls.ShowImage([im_tophat, im_blackhat, im_fillhole, im_skeleton, im_convexhull], 1, 5, 
                ImageTitleList= ["Tophat", "BlackHat", "FillHole", "Skeleton", "Convexhull"])
      
    result = result.astype(bool)
    return result, image_mask_color

  @classmethod
  def ConcatImage(cls,ImageList, CombineType = "h", display = 1):
    """
      Concatenate Images

      Parameters
      ----------
      ImageList : list
          The list of images to be concatenated
      CombineType : string
          The type of concatenation
      display : int
          Display the concatenated image or not

      Returns
      -------
      result : numpy array
          The concatenated image
    """
    if(CombineType == "h"):
      result = np.concatenate(ImageList, axis=1)
    else:
      result = np.concatenate(ImageList, axis=0)
    
    if(display):
      cls.ShowImage([result], 1, 1)

    result = np.uint8(result)
    return result

  @classmethod   
  def GenerateOutput(cls,image, image_mask, FilePath = "", SaveFolderPath = ""):
    """
      Generate Output 

      Parameters
      ----------
      image : numpy array
          The image to be transformed
      image_mask : numpy array
          The image mask
      FilePath : string
          The path of the image
      SaveFolderPath : string
          The path of the folder to save the output

      Returns
      -------
      result : numpy array
          The transformed image
    
    """
    image_output_rgbmask = cv2.merge([image_mask*255, image_mask*255, image_mask*255])
    image_output_fill = cls.LabelObjectByMask(image, image_mask, type = "Fill", color = (255,0,0))
    image_output_bbox = cls.LabelObjectByMask(image, image_mask, type = "BBox", color = (255,0,0))
    image_output_boundary = cls.LabelObjectByMask(image, image_mask, type = "Boundary", color = (255,0,0))

    DemoImage1 = cls.ConcatImage([image_output_rgbmask, image_output_fill], display= 0)
    DemoImage2 = cls.ConcatImage([image_output_bbox, image_output_boundary], display= 0)
    DemoImage = cls.ConcatImage([DemoImage1, DemoImage2], CombineType = "v", display= 0)

    if(len(SaveFolderPath) > 0):
      filename = Path(FilePath).stem
      print(f"Already Save Results to Folder {SaveFolderPath}")
      
      print(f'{SaveFolderPath}/{filename}_rgbmask.jpg')
      cv2.imwrite(f'{SaveFolderPath}/{filename}_mask.jpg', image_output_rgbmask)

      print(f'{SaveFolderPath}/{filename}_demo.jpg')
      cv2.imwrite(f'{SaveFolderPath}/{filename}_demo.jpg', cv2.cvtColor(DemoImage, cv2.COLOR_RGB2BGR))

    return image_output_rgbmask, image_output_fill, image_output_bbox, image_output_boundary, DemoImage
  @classmethod
  def doCoconutMaskSegment(cls,AnImage, image_mask, display = 1):
    MaskList = []
    MaskList.append(image_mask)
    image_mask_adjust = cls.Morphology(AnImage, image_mask, morType = "fillhole", size = 8, display = 0)
    MaskList.append(image_mask_adjust)
    image_mask_adjust = cls.Morphology(AnImage, image_mask_adjust, morType = "erosion", size = 8, display = 0)
    MaskList.append(image_mask_adjust)
    image_mask_adjust = cls.GetLargestBinaryArea(AnImage, image_mask_adjust, display = 0)
    MaskList.append(image_mask_adjust)
    image_mask_adjust = cls.Morphology(AnImage, image_mask_adjust, morType = "dilation", size = 10, display = 0)
    MaskList.append(image_mask_adjust)
    
    if(display):
      cls.ShowImage(MaskList, 1, 5)
    
    return image_mask_adjust
  @classmethod
  def doCoconutMaskSegment2(cls,AnImage, image_mask, display = 1):
    MaskList = []
    MaskList.append(image_mask)
    image_mask_adjust = cls.Morphology(AnImage, image_mask, morType = "fillhole", size = 8, display = 0)
    MaskList.append(image_mask_adjust)
    image_mask_adjust = cls.Morphology(AnImage, image_mask_adjust, morType = "erosion", size = 50, display = 0)
    MaskList.append(image_mask_adjust)
    image_mask_adjust = cls.GetLargestBinaryArea(AnImage, image_mask_adjust, display = 0)
    MaskList.append(image_mask_adjust)
    image_mask_adjust = cls.Morphology(AnImage, image_mask_adjust, morType = "dilation", size = 50, display = 0)
    MaskList.append(image_mask_adjust)
    
    if(display):
      cls.ShowImage(MaskList, 1, 5)
    
    return image_mask_adjust