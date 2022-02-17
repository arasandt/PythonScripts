from collections import namedtuple

from PIL import Image, ImageDraw

from scipy.spatial import distance
from collections import Counter
import matplotlib.pyplot as plt
from operator import itemgetter
from mpl_toolkits.mplot3d import Axes3D
import cv2, math
from matplotlib import cm
from matplotlib import colors
import numpy as np
from math import sqrt
import random

all_colors = {'Pink_Pink' : (255,192,203),
        'Pink_LightPink' : (255,182,193),
        'Pink_HotPink' : (255,105,180),
        'Pink_DeepPink' : (255,20,147),
        'Pink_PaleVioletRed' : (219,112,147),
        'Pink_MediumVioletRed' : (199,21,133),
        'Red_LightSalmon' : (255,160,122),
        'Red_Salmon' : (250,128,114),
        'Red_DarkSalmon' : (233,150,122),
        'Red_LightCoral' : (240,128,128),
        'Red_IndianRed' : (205,92,92),
        'Red_Crimson' : (220,20,60),
        'Red_Firebrick' : (178,34,34),
        'Red_DarkRed' : (139,0,0),
        'Red_Red' : (255,0,0),
        'Orange_OrangeRed' : (255,69,0),
        'Orange_Tomato' : (255,99,71),
        'Orange_Coral' : (255,127,80),
        'Orange_DarkOrange' : (255,140,0),
        'Orange_Orange' : (255,165,0),
        'Yellow_Yellow' : (255,255,0),
        'Yellow_LightYellow' : (255,255,224),
        'Yellow_LemonChiffon' : (255,250,205),
        'Yellow_LightGoldenrodYellow ' : (250,250,210),
        'Yellow_PapayaWhip' : (255,239,213),
        'Yellow_Moccasin' : (255,228,181),
        'Yellow_PeachPuff' : (255,218,185),
        'Yellow_PaleGoldenrod' : (238,232,170),
        'Yellow_Khaki' : (240,230,140),
        'Yellow_DarkKhaki' : (189,183,107),
        'Yellow_Gold' : (255,215,0),
        'Brown_Cornsilk' : (255,248,220),
        'Brown_BlanchedAlmond' : (255,235,205),
        'Brown_Bisque' : (255,228,196),
        'Brown_NavajoWhite' : (255,222,173),
        'Brown_Wheat' : (245,222,179),
        'Brown_Burlywood' : (222,184,135),
        'Brown_Tan' : (210,180,140),
        'Brown_RosyBrown' : (188,143,143),
        'Brown_SandyBrown' : (244,164,96),
        'Brown_Goldenrod' : (218,165,32),
        'Brown_DarkGoldenrod' : (184,134,11),
        'Brown_Peru' : (205,133,63),
        'Brown_Chocolate' : (210,105,30),
        'Brown_SaddleBrown' : (139,69,19),
        'Brown_Sienna' : (160,82,45),
        'Brown_Brown' : (165,42,42),
        'Brown_Maroon' : (128,0,0),
        'Green_DarkOliveGreen' : (85,107,47),
        'Green_Olive' : (128,128,0),
        'Green_OliveDrab' : (107,142,35),
        'Green_YellowGreen' : (154,205,50),
        'Green_LimeGreen' : (50,205,50),
        'Green_Lime' : (0,255,0),
        'Green_LawnGreen' : (124,252,0),
        'Green_Chartreuse' : (127,255,0),
        'Green_GreenYellow' : (173,255,47),
        'Green_SpringGreen' : (0,255,127),
        'Green_MediumSpringGreen ' : (0,250,154),
        'Green_LightGreen' : (144,238,144),
        'Green_PaleGreen' : (152,251,152),
        'Green_DarkSeaGreen' : (143,188,143),
        'Green_MediumAquamarine' : (102,205,170),
        'Green_MediumSeaGreen' : (60,179,113),
        'Green_SeaGreen' : (46,139,87),
        'Green_ForestGreen' : (34,139,34),
        'Green_Green' : (0,128,0),
        'Green_DarkGreen' : (0,100,0),
        'Cyan_Aqua' : (0,255,255),
        'Cyan_Cyan' : (0,255,255),
        'Cyan_LightCyan' : (224,255,255),
        'Cyan_PaleTurquoise' : (175,238,238),
        'Cyan_Aquamarine' : (127,255,212),
        'Cyan_Turquoise' : (64,224,208),
        'Cyan_MediumTurquoise' : (72,209,204),
        'Cyan_DarkTurquoise' : (0,206,209),
        'Cyan_LightSeaGreen' : (32,178,170),
        'Cyan_CadetBlue' : (95,158,160),
        'Cyan_DarkCyan' : (0,139,139),
        'Cyan_Teal' : (0,128,128),
        'Blue_LightSteelBlue' : (176,196,222),
        'Blue_PowderBlue' : (176,224,230),
        'Blue_LightBlue' : (173,216,230),
        'Blue_SkyBlue' : (135,206,235),
        'Blue_LightSkyBlue' : (135,206,250),
        'Blue_DeepSkyBlue' : (0,191,255),
        'Blue_DodgerBlue' : (30,144,255),
        'Blue_CornflowerBlue' : (100,149,237),
        'Blue_SteelBlue' : (70,130,180),
        'Blue_RoyalBlue' : (65,105,225),
        'Blue_Blue' : (0,0,255),
        'Blue_MediumBlue' : (0,0,205),
        'Blue_DarkBlue' : (0,0,139),
        'Blue_Navy' : (0,0,128),
        'Blue_MidnightBlue' : (25,25,112),
        'Purple_Lavender' : (230,230,250),
        'Purple_Thistle' : (216,191,216),
        'Purple_Plum' : (221,160,221),
        'Purple_Violet' : (238,130,238),
        'Purple_Orchid' : (218,112,214),
        'Purple_Fuchsia' : (255,0,255),
        'Purple_Magenta' : (255,0,255),
        'Purple_MediumOrchid' : (186,85,211),
        'Purple_MediumPurple' : (147,112,219),
        'Purple_BlueViolet' : (138,43,226),
        'Purple_DarkViolet' : (148,0,211),
        'Purple_DarkOrchid' : (153,50,204),
        'Purple_DarkMagenta' : (139,0,139),
        'Purple_Purple' : (128,0,128),
        'Purple_Indigo' : (75,0,130),
        'Purple_DarkSlateBlue' : (72,61,139),
        'Purple_SlateBlue' : (106,90,205),
        'Purple_MediumSlateBlue ' : (123,104,238),
        'White_White' : (255,255,255),
        'White_Snow' : (255,250,250),
        'White_Honeydew' : (240,255,240),
        'White_MintCream' : (245,255,250),
        'White_Azure' : (240,255,255),
        'White_AliceBlue' : (240,248,255),
        'White_GhostWhite' : (248,248,255),
        'White_WhiteSmoke' : (245,245,245),
        'White_Seashell' : (255,245,238),
        'White_Beige' : (245,245,220),
        'White_OldLace' : (253,245,230),
        'White_FloralWhite' : (255,250,240),
        'White_Ivory' : (255,255,240),
        'White_AntiqueWhite' : (250,235,215),
        'White_Linen' : (250,240,230),
        'White_LavenderBlush' : (255,240,245),
        'White_MistyRose' : (255,228,225),
        'Gray_Gainsboro' : (220,220,220),
        'Gray_LightGray' : (211,211,211),
        'Gray_Silver' : (192,192,192),
        'Gray_DarkGray' : (169,169,169),
        'Gray_Gray' : (128,128,128),
        'Gray_DimGray' : (105,105,105),
        'Gray_LightSlateGray' : (119,136,153),
        'Gray_SlateGray' : (112,128,144),
        'Gray_DarkSlateGray' : (47,79,79),
        #'Black_Black' : (0,0,0),
        }
    

try:
    import Image
except ImportError:
    from PIL import Image

Point = namedtuple('Point', ('coords', 'n', 'ct'))
Cluster = namedtuple('Cluster', ('points', 'center', 'n'))

def get_colors(infile, outfile, numcolors=10, swatchsize=20, resize=150):

    image = Image.open(infile)
    image = image.resize((resize, resize))
    result = image.convert('P', palette=Image.ADAPTIVE, colors=numcolors)
    result.putalpha(0)
    colors = result.getcolors(resize*resize)

    # Save colors to file

    pal = Image.new('RGB', (swatchsize*numcolors, swatchsize))
    draw = ImageDraw.Draw(pal)

    posx = 0
    for count, col in colors:
        draw.rectangle([posx, 0, posx+swatchsize, swatchsize], fill=col)
        posx = posx + swatchsize

    del draw
    pal.save(outfile, "PNG")


def get_points(img):
    points = []
    w, h = img.size
    for count, color in img.getcolors(w * h):
        points.append(Point(color, 3, count))
    return points

rtoh = lambda rgb: '#%s' % ''.join(('%02x' % p for p in rgb))

def colorz(filename, n=1):
    print(filename)
    img = Image.open(filename)
    cv2.imshow('he1',cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))
    width, height = img.size
    new_width = width // 2
    new_height = height // 2
    left = (width - new_width)//2
    top = (height - new_height)//2
    right = (width + new_width)//2
    bottom = (height + new_height)//2
    img = img.crop((left, top, right, bottom))
    cv2.imshow('he',cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    w, h = img.size
# =============================================================================
# 
#     flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
#     nemo = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
#     #plt.imshow(nemo)
#     #plt.show()
#     
#     r, g, b = cv2.split(nemo)
#     fig = plt.figure()
#     axis = fig.add_subplot(1, 1, 1, projection="3d")
#     pixel_colors = nemo.reshape((np.shape(nemo)[0]*np.shape(nemo)[1], 3))
#     norm = colors.Normalize(vmin=-1.,vmax=1.)
#     norm.autoscale(pixel_colors)
#     pixel_colors = norm(pixel_colors).tolist()
#     axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
#     axis.set_xlabel("Red")
#     axis.set_ylabel("Green")
#     axis.set_zlabel("Blue")
#     #plt.show()
#     
#     hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)    
#     h, s, v = cv2.split(hsv_nemo)
#     fig = plt.figure()
#     axis = fig.add_subplot(1, 1, 1, projection="3d")
#     
#     axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
#     axis.set_xlabel("Hue")
#     axis.set_ylabel("Saturation")
#     axis.set_zlabel("Value")
#     #plt.show()    
# =============================================================================
    
    
    
    
    
    
    #img.thumbnail((200, 200))
    #w, h = img.size
    #img_arr = np.asarray(img)
    #all_rgb_codes = img_arr.reshape(-1, img_arr.shape[-1])
    #unique_rgbs = np.unique(all_rgb_codes, axis=0, return_counts = True)
    #print(unique_rgbs)
    points = get_points(img)
    clusters = kmeans(points, n, 2)
    rgbs = [map(int, c.center.coords) for c in clusters]
    
    #return map(rtoh, rgbs)
    return map(list, rgbs)

def euclidean(p1, p2):
    return sqrt(sum([
        (p1.coords[i] - p2.coords[i]) ** 2 for i in range(p1.n)
    ]))

def calculate_center(points, n):
    vals = [0.0 for i in range(n)]
    plen = 0
    for p in points:
        plen += p.ct
        for i in range(n):
            vals[i] += (p.coords[i] * p.ct)
    return Point([(v / plen) for v in vals], n, 1)

def kmeans(points, k, min_diff):
    clusters = [Cluster([p], p, p.n) for p in random.sample(points, k)]

    while 1:
        plists = [[] for i in range(k)]

        for p in points:
            smallest_distance = float('Inf')
            for i in range(k):
                distance = euclidean(p, clusters[i].center)
                if distance < smallest_distance:
                    smallest_distance = distance
                    idx = i
            plists[idx].append(p)

        diff = 0
        for i in range(k):
            old = clusters[i]
            center = calculate_center(plists[i], old.n)
            new = Cluster(plists[i], center, old.n)
            clusters[i] = new
            diff = max(diff, euclidean(old.center, new.center))

        if diff < min_diff:
            break

    return clusters

def dcolor(infile):
    import cv2
    import numpy as np
    #import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    
    
    def find_histogram(clt):
        """
        create a histogram with k clusters
        :param: clt
        :return:hist
        """
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    
        hist = hist.astype("float")
        hist /= hist.sum()
    
        return hist
    
    def plot_colors2(hist, centroids):
        bar = np.zeros((50, 300, 3), dtype="uint8")
        startX = 0
        colors_with_percent = []
        for (percent, color) in zip(hist, centroids):
            # plot the relative percentage of each cluster
            rgb_triplet = color.astype("uint8").tolist()
            colors_with_percent.append((round(percent*100,2), rgb_triplet))
            #print(rgb_triplet, percent)
            endX = startX + (percent * 300)
            cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                          rgb_triplet, -1)
            startX = endX
    
        # return the bar chart
        return bar, colors_with_percent
    
    img = cv2.imread(infile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
    clt = KMeans(n_clusters=3) #cluster number
    clt.fit(img)
    
    hist = find_histogram(clt)
    bar, rgb_triplet = plot_colors2(hist, clt.cluster_centers_)
    
    #plt.axis("off")
    #plt.imshow(bar)
    #plt.show()
    return rgb_triplet
     

def color_extractor(infile):
    import cv2
    import numpy as np
    
    from color_extractor import ImageToColor
    
    npz = np.load('color_names.npz')
    img_to_color = ImageToColor(npz['samples'], npz['labels'])
    
    img = cv2.imread(infile)
    print(img_to_color.get(img))    
    
def ML_Color(infile):
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    from collections import Counter
    from skimage.color import rgb2lab, deltaE_cie76
    import os
    #image = cv2.imread(infile)
    #print("The type of this input is {}".format(type(image)))
    #print("Shape: {}".format(image.shape))
    #plt.imshow(image)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #plt.imshow(image)
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #plt.imshow(gray_image, cmap='gray')
    #resized_image = cv2.resize(image, (1200, 600))
    #plt.imshow(resized_image)
    def RGB2HEX(color):
        return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))
    def get_image(image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    def get_colors(image, number_of_colors, show_chart):
        
        modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
        modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
        
        clf = KMeans(n_clusters = number_of_colors)
        labels = clf.fit_predict(modified_image)
        
        counts = Counter(labels)
        
        center_colors = clf.cluster_centers_
        # We get ordered colors by iterating through the keys
        ordered_colors = [center_colors[i]/255 for i in counts.keys()]
        #hex_colors = [RGB2HEX(ordered_colors[i]*255) for i in counts.keys()]
        rgb_colors = [ordered_colors[i]*255 for i in counts.keys()]
        rgb_colors = [list(map(int,i)) for i in rgb_colors]
        #rgb_colors = list(map)
        
        #if (show_chart):
        #    plt.figure(figsize = (8, 6))
        #    plt.pie(counts.values(), labels = hex_colors, colors = ordered_colors)
        
        return rgb_colors
        #return rgb_colors.astype(np.int)
    
    print(get_colors(get_image(infile), 3, True))
    return
    IMAGE_DIRECTORY = 'images'
    COLORS = {
        'GREEN': [0, 128, 0],
        'BLUE': [0, 0, 128],
        'YELLOW': [255, 255, 0]
    }
    images = []
    
    for file in os.listdir(IMAGE_DIRECTORY):
        if not file.startswith('.'):
            images.append(get_image(os.path.join(IMAGE_DIRECTORY, file)))
    plt.figure(figsize=(20, 10))
    for i in range(len(images)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(images[i])
    def match_image_by_color(image, color, threshold = 60, number_of_colors = 10): 
        
        image_colors = get_colors(image, number_of_colors, False)
        selected_color = rgb2lab(np.uint8(np.asarray([[color]])))
    
        select_image = False
        for i in range(number_of_colors):
            curr_color = rgb2lab(np.uint8(np.asarray([[image_colors[i]]])))
            diff = deltaE_cie76(selected_color, curr_color)
            if (diff < threshold):
                select_image = True
        
        return select_image
    def show_selected_images(images, color, threshold, colors_to_match):
        index = 1
        
        for i in range(len(images)):
            selected = match_image_by_color(images[i],
                                            color,
                                            threshold,
                                            colors_to_match)
            if (selected):
                plt.subplot(1, 5, index)
                plt.imshow(images[i])
                index += 1
    plt.figure(figsize = (20, 10))
    show_selected_images(images, COLORS['GREEN'], 60, 5)
    plt.figure(figsize = (20, 10))
    show_selected_images(images, COLORS['BLUE'], 60, 5)
    plt.figure(figsize = (20, 10))
    show_selected_images(images, COLORS['YELLOW'], 60, 5)            
    

    

if __name__ == '__main__':

    infile = './output/F2_f2775_p0.jpg'
    #ML_Color(infile)
    colors_with_percent = dcolor(infile)
    colors_with_percent = sorted(colors_with_percent, key=itemgetter(0), reverse=True)
    print(colors_with_percent)
#    all_colors = {'PinkPink':(255,192,203),
#            'red':(255, 0, 0),
#                  'green': (0,255,0),
#                  'blue': (0,0,255),
#                  'white': (255,255,255),
#                  'black': (0,0,0),
#                  }
    #colors_with_percent = cv2.imread(infile).flatten()
    #col = []
    
    #for percent, fcolor in enumerate(colors_with_percent):
    #for percent, fcolor in colors_with_percent:
    #    #print(all_colors.values())
    #    print(percent,len(colors_with_percent))
    #    dismat = [(i, distance.euclidean(fcolor,j)) for i, j in all_colors.items()]
    #    #print(dismat)
    #    col.append(sorted(dismat, key=itemgetter(1))[0][0])
    #print(Counter(col))

    #color_extractor(infile)
    #get_colors(infile, 'outfile.png')
    
    #all_colors = colorz(infile)
#    import webcolors
#    
#    def get_colour_name(rgb_triplet):
#        min_colours = {}
#        for key, name in webcolors.css21_hex_to_names.items():
#            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
#            rd = (r_c - rgb_triplet[0]) ** 2
#            gd = (g_c - rgb_triplet[1]) ** 2
#            bd = (b_c - rgb_triplet[2]) ** 2
#            min_colours[(rd + gd + bd)] = name
#        return min_colours[min(min_colours.keys())]    
#    
    
    #for i in all_colors:
    #    print(i,list(map(rtoh,[i]))[0])
        #print(get_colour_name(i))
    
    #cv2.destroyAllWindows()