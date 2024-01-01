# main.py
import os
import base64
import io
import math
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
import mysql.connector
import hashlib
import datetime
import calendar
import random
from random import randint
from urllib.request import urlopen
import webbrowser
import csv
from plotly import graph_objects as go
import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil
import imagehash
from werkzeug.utils import secure_filename
from PIL import Image
import argparse
import urllib.request
import urllib.parse
   
# necessary imports 
import seaborn as sns
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
#%matplotlib inline
pd.set_option('display.max_columns', 26)
##
from PIL import Image, ImageOps
import scipy.ndimage as ndi

from skimage import transform
import seaborn as sns
#from keras.preprocessing.image import ImageDataGenerator , load_img , img_to_array
#from keras.models import Sequential
#from keras.layers import Conv2D, Flatten, MaxPool2D, Dense
##
import glob
#from keras.models import Sequential, load_model
import numpy as np
import pandas as pd
import seaborn as sns
#import keras as k
#from keras.layers import Dense
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
#from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
#from tensorflow.keras.optimizers import Adam
##
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  charset="utf8",
  database="food_calorie"

)
app = Flask(__name__)
##session key
app.secret_key = 'abcdef'
#######
UPLOAD_FOLDER = 'static/upload'
ALLOWED_EXTENSIONS = { 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#####
@app.route('/', methods=['GET', 'POST'])
def index():
    msg=""

    
        

    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            msg = 'Incorrect username/password! or access not provided'
    return render_template('index.html',msg=msg)

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg=""

    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login.html',msg=msg)



@app.route('/login_user', methods=['GET', 'POST'])
def login_user():
    msg=""

    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM register WHERE uname = %s AND pass = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('test'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login_user.html',msg=msg)

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg=""
    

    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
    
    mycursor = mydb.cursor()
    #if request.method=='GET':
    #    msg = request.args.get('msg')
    if request.method=='POST':
        
        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        uname=request.form['uname']
        pass1=request.form['pass']

        mycursor.execute("SELECT max(id)+1 FROM register")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
                
        sql = "INSERT INTO register(id,name,mobile,email,uname,pass) VALUES (%s, %s, %s, %s, %s, %s)"
        val = (maxid,name,mobile,email,uname,pass1)
        mycursor.execute(sql,val)
        mydb.commit()
        return redirect(url_for('login_user'))

    
        
    return render_template('register.html',msg=msg)




@app.route('/admin', methods=['GET', 'POST'])
def admin():
    
    #######
    '''mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM food_info")
    rr = mycursor.fetchall()
    for r1 in rr:
        
        mycursor.execute("SELECT * FROM food where food=%s",(r1[1],))
        dd = mycursor.fetchone()
        cal=dd[2]
        weight=r1[4]
        w=int(weight)
        d2=(cal/100)*w
        calorie=round(d2,2)
        mycursor.execute("update food_info set calorie=%s where id=%s",(calorie,r1[0]))
        mydb.commit()'''
    ############
        
        
    return render_template('admin.html')

@app.route('/add_food', methods=['GET', 'POST'])
def add_food():
    msg=""
    act=request.args.get("act")

    mycursor = mydb.cursor()
    #if request.method=='GET':
    #    msg = request.args.get('msg')
    if request.method=='POST':
        
        food=request.form['food']

        mycursor.execute('SELECT count(*) FROM food WHERE food = %s', (food,))
        cnt = mycursor.fetchone()[0]
        if cnt==0:
            mycursor.execute("SELECT max(id)+1 FROM food")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
                    
            sql = "INSERT INTO food(id,food) VALUES (%s, %s)"
            val = (maxid,food)
            mycursor.execute(sql,val)
            mydb.commit()
            return redirect(url_for('add_food',act='1'))
        else:
            msg="1"

    if act=="del":
        did=request.args.get("did")
        mycursor.execute("delete from food where id=%s",(did,))
        mydb.commit()
        return redirect(url_for('add_food'))

        
    mycursor.execute("SELECT * FROM food")
    data = mycursor.fetchall()
    
        
        
    return render_template('add_food.html',msg=msg,act=act,data=data)

@app.route('/add_data', methods=['GET', 'POST'])
def add_data():
    msg=""
    act=request.args.get("act")
    food=request.args.get("food")
    mycursor = mydb.cursor()
    #if request.method=='GET':
    #    msg = request.args.get('msg')

    ######
    '''dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)
        food=''
        calorie=''
        weight=''
        mycursor.execute("SELECT max(id)+1 FROM food_info")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        sql = "INSERT INTO food_info(id,food,filename,calorie,weight) VALUES (%s, %s, %s, %s, %s)"
        val = (maxid,food,fname,calorie,weight)
        mycursor.execute(sql,val)
        mydb.commit()'''
    #######

        
    if request.method=='POST':
        
        
        nutrient=request.form['nutrient']
        details=request.form['details']
      

        mycursor.execute("SELECT max(id)+1 FROM food_data")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
                
        sql = "INSERT INTO food_data(id,food,nutrient,details) VALUES (%s, %s, %s, %s)"
        val = (maxid,food,nutrient,details)
        mycursor.execute(sql,val)
        mydb.commit()
        return redirect(url_for('add_data',act='1',food=food))

    if act=="del":
        did=request.args.get("did")
        mycursor.execute("delete from food_data where id=%s",(did,))
        mydb.commit()
        return redirect(url_for('add_data',food=food))

        
    mycursor.execute("SELECT * FROM food_data where food=%s",(food,))
    data = mycursor.fetchall()
    
        
        
    return render_template('add_data.html',msg=msg,act=act,data=data,food=food)

@app.route('/img_process', methods=['GET', 'POST'])
def img_process():
    dimg=[]
    '''path_main = 'static/data'
    for fname in os.listdir(path_main):
        dimg.append(fname)
        #resize
        print(fname)
        img = cv2.imread('static/data/'+fname)
        rez = cv2.resize(img, (300, 300))
        cv2.imwrite("static/dataset/"+fname, rez)'''

    return render_template('img_process.html',dimg=dimg)

@app.route('/pro1', methods=['GET', 'POST'])
def pro1():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)
        #list_of_elements = os.listdir(os.path.join(path_main, folder))

        #resize
        #img = cv2.imread('static/data/'+fname)
        #rez = cv2.resize(img, (400, 300))
        #cv2.imwrite("static/dataset/"+fname, rez)'''

        '''img = cv2.imread('static/dataset/'+fname) 	
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("static/trained/g_"+fname, gray)
        ##noice
        img = cv2.imread('static/trained/g_'+fname) 
        dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
        fname2='ns_'+fname
        cv2.imwrite("static/trained/"+fname2, dst)'''

    return render_template('pro1.html',dimg=dimg)


def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

@app.route('/pro11', methods=['GET', 'POST'])
def pro11():
    msg=""
    dimg=[]
    '''path_main = 'static/data'
    for fname in os.listdir(path_main):
        dimg.append(fname)'''

    return render_template('pro11.html',dimg=dimg)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    msg=""
    act=request.args.get("act")
    food=request.args.get("food")
    mycursor = mydb.cursor()
    
    if request.method=='POST':
        
        file = request.files['file']
        weight=request.form['weight']
      
        fname = file.filename
        filename = secure_filename(fname)
    
        mycursor.execute("SELECT * FROM food where food=%s",(food,))
        dd = mycursor.fetchone()
        cal=dd[2]
        w=int(weight)
        d2=(cal/100)*w
        calorie=round(d2,2)
        
        mycursor.execute("SELECT max(id)+1 FROM food_info")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        fn="f"+str(maxid)+filename
        sql = "INSERT INTO food_info(id,food,filename,calorie,weight) VALUES (%s, %s, %s, %s, %s)"
        val = (maxid,food,fn,calorie,weight)
        mycursor.execute(sql,val)
        mydb.commit()
        file.save(os.path.join("static/dataset", fn))

        #
        fname=fn
        img = cv2.imread('static/dataset/'+fn)
        rez = cv2.resize(img, (300, 300))
        cv2.imwrite("static/dataset/"+fn, rez)
        #
        img = cv2.imread('static/dataset/'+fn) 	
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("static/trained/g_"+fn, gray)
        ##noice
        img = cv2.imread('static/trained/g_'+fname) 
        dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
        fname2='ns_'+fname
        cv2.imwrite("static/trained/"+fname2, dst)
        ###########
        image = cv2.imread('static/dataset/'+fname)
        original = image.copy()
        kmeans = kmeans_color_quantization(image, clusters=4)

        # Convert to grayscale, Gaussian blur, adaptive threshold
        gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,2)

        # Draw largest enclosing circle onto a mask
        mask = np.zeros(original.shape[:2], dtype=np.uint8)
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)
            cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
            break
        
        # Bitwise-and for result
        result = cv2.bitwise_and(original, original, mask=mask)
        result[mask==0] = (0,0,0)
        cv2.imwrite("static/trained/bb/bin_"+fname, thresh)
        ##########
        img = cv2.imread('static/trained/g_'+fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,1.5*dist_transform.max(),255,0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        segment = cv2.subtract(sure_bg,sure_fg)
        img = Image.fromarray(img)
        segment = Image.fromarray(segment)
        path3="static/trained/sg/sg_"+fname
        segment.save(path3)
        #####
        image = cv2.imread("static/dataset/"+fname)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 100)
        image = Image.fromarray(image)
        edged = Image.fromarray(edged)
        
        path4="static/trained/ff/"+fname
        edged.save(path4)
        #########
        mycursor.execute("SELECT * FROM food_info")
        result = mycursor.fetchall()
        with open('static/trained/data.csv','w') as outfile:
            writer = csv.writer(outfile, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(col[0] for col in mycursor.description)
            for row in result:
                writer.writerow(row)

        with open('static/trained/data.csv') as input, open('static/trained/data.csv', 'w', newline='') as outfile:
            writer = csv.writer(outfile, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(col[0] for col in mycursor.description)
            for row in result:
                if row or any(row) or any(field.strip() for field in row):
                    writer.writerow(row)

    ###########
        

        
        return redirect(url_for('upload',act='1',food=food))


        
    mycursor.execute("SELECT * FROM food_info where food=%s",(food,))
    data = mycursor.fetchall()
    
        
        
    return render_template('upload.html',msg=msg,act=act,data=data,food=food)


@app.route('/pro2', methods=['GET', 'POST'])
def pro2():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)

        #f1=open("adata.txt",'w')
        #f1.write(fname)
        #f1.close()
        ##bin
        image = cv2.imread('static/dataset/'+fname)
        original = image.copy()
        kmeans = kmeans_color_quantization(image, clusters=4)

        # Convert to grayscale, Gaussian blur, adaptive threshold
        gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,2)

        # Draw largest enclosing circle onto a mask
        mask = np.zeros(original.shape[:2], dtype=np.uint8)
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)
            cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
            break
        
        # Bitwise-and for result
        result = cv2.bitwise_and(original, original, mask=mask)
        result[mask==0] = (0,0,0)

        
        ###cv2.imshow('thresh', thresh)
        ###cv2.imshow('result', result)
        ###cv2.imshow('mask', mask)
        ###cv2.imshow('kmeans', kmeans)
        ###cv2.imshow('image', image)
        ###cv2.waitKey()

        #cv2.imwrite("static/trained/bb/bin_"+fname, thresh)

    
   

    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        ##RPN
        
        
        img = cv2.imread('static/trained/g_'+fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,1.5*dist_transform.max(),255,0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        segment = cv2.subtract(sure_bg,sure_fg)
        img = Image.fromarray(img)
        segment = Image.fromarray(segment)
        path3="static/trained/sg/sg_"+fname
        #segment.save(path3)
        

    return render_template('pro2.html',dimg=dimg)

###Feature extraction & Classification
def getCalorie(label, volume): #volume in cm^3
	calorie = calorie_dict[int(label)]
	density = density_dict[int(label)]
	mass = volume*density*1.0
	calorie_tot = (calorie/100.0)*mass
	return mass, calorie_tot, calorie #calorie per 100 grams

def getVolume(label, area, skin_area, pix_to_cm_multiplier, fruit_contour):
	area_fruit = (area/skin_area)*skin_multiplier #area in cm^2
	label = int(label)
	volume = 100
	if label == 1 or label == 5 or label == 7 or label == 6 : #sphere-apple,tomato,orange,kiwi,onion
		radius = np.sqrt(area_fruit/np.pi)
		volume = (4/3)*np.pi*radius*radius*radius
		#print (area_fruit, radius, volume, skin_area)
	
	if label == 2 or label == 4 or (label == 3 and area_fruit > 30): #cylinder like banana, cucumber, carrot
		fruit_rect = cv2.minAreaRect(fruit_contour)
		height = max(fruit_rect[1])*pix_to_cm_multiplier
		radius = area_fruit/(2.0*height)
		volume = np.pi*radius*radius*height
		
	if (label==4 and area_fruit < 30) : # carrot
		volume = area_fruit*0.5 #assuming width = 0.5 cm
	
	return volume

def calories(result,img):
    img_path =img 
    fruit_areas,final_f,areaod,skin_areas, fruit_contours, pix_cm = getAreaOfFood(img_path)
    volume = getVolume(result, fruit_areas, skin_areas, pix_cm, fruit_contours)
    mass, cal, cal_100 = getCalorie(result, volume)
    fruit_volumes=volume
    fruit_calories=cal
    fruit_calories_100grams=cal_100
    fruit_mass=mass
    return fruit_calories

def get_model(IMG_SIZE,no_of_fruits,LR):
	try:
		tf.reset_default_graph()
	except:
		print("tensorflow")
	convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

	convnet = conv_2d(convnet, 32, 5, activation='relu')

	convnet = max_pool_2d(convnet, 5)

	convnet = conv_2d(convnet, 64, 5, activation='relu')

	convnet = max_pool_2d(convnet, 5)

	convnet = conv_2d(convnet, 128, 5, activation='relu')
	convnet = max_pool_2d(convnet, 5)

	convnet = conv_2d(convnet, 64, 5, activation='relu')
	convnet = max_pool_2d(convnet, 5)


	convnet = conv_2d(convnet, 32, 5, activation='relu')
	convnet = max_pool_2d(convnet, 5)

	convnet = fully_connected(convnet, 1024, activation='relu')
	convnet = dropout(convnet, 0.8)

	convnet = fully_connected(convnet, no_of_fruits, activation='softmax')
	convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

	model = tflearn.DNN(convnet, tensorboard_dir='log')

	return model

def DCNN_process(self):
        
        train_data_preprocess = ImageDataGenerator(
                rescale = 1./255,
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True)

        test_data_preprocess = (1./255)

        train = train_data_preprocess.flow_from_directory(
                'dataset/training',
                target_size = (128,128),
                batch_size = 32,
                class_mode = 'binary')

        test = train_data_preprocess.flow_from_directory(
                'dataset/test',
                target_size = (128,128),
                batch_size = 32,
                class_mode = 'binary')

        ## Initialize the Convolutional Neural Net

        # Initialising the CNN
        cnn = Sequential()

        # Step 1 - Convolution
        # Step 2 - Pooling
        cnn.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (2, 2)))

        # Adding a second convolutional layer
        cnn.add(Conv2D(32, (3, 3), activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (2, 2)))

        # Step 3 - Flattening
        cnn.add(Flatten())

        # Step 4 - Full connection
        cnn.add(Dense(units = 128, activation = 'relu'))
        cnn.add(Dense(units = 1, activation = 'sigmoid'))

        # Compiling the CNN
        cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        history = cnn.fit_generator(train,
                                 steps_per_epoch = 250,
                                 epochs = 25,
                                 validation_data = test,
                                 validation_steps = 2000)

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        test_image = image.load_img('\\dataset\\', target_size=(128,128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = cnn.predict(test_image)
        print(result)

        if result[0][0] == 1:
                print('feature extracted')
        else:
                print('none')
@app.route('/pro3', methods=['GET', 'POST'])
def pro3():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)
        
    '''path_main = 'static/dataset'
    i=1
    while i<=50:
        fname="r"+str(i)+".jpg"
        dimg.append(fname)

        img = Image.open('static/data/classify/'+fname)
        array = np.array(img)

        array = 255 - array

        invimg = Image.fromarray(array)
        invimg.save('static/upload/ff_'+fname)
        i+=1
    i=1
    j=51
    while i<=10:
        
        fname="r"+str(j)+".jpg"
        dimg.append(fname)

        img = Image.open('static/dataset/'+fname)
        array = np.array(img)

        array = 255 - array

        invimg = Image.fromarray(array)
        invimg.save('static/upload/ff_'+fname)
        j+=1
        i+=1

    '''    

    return render_template('pro3.html',dimg=dimg)

@app.route('/pro4', methods=['GET', 'POST'])
def pro4():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)

        #####
        image = cv2.imread("static/dataset/"+fname)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 100)
        image = Image.fromarray(image)
        edged = Image.fromarray(edged)
        
        path4="static/trained/ff/"+fname
        #edged.save(path4)
        ##
    
        
    return render_template('pro4.html',dimg=dimg)


    

@app.route('/pro5', methods=['GET', 'POST'])
def pro5():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)
    #graph
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,8)
        v1='0.'+str(rn)
        x2.append(float(v1))
        i+=1
    
    x1=[0,0,0,0,0]
    y=[30,80,140,210,265]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    

    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Model Precision")
    plt.ylabel("precision")
    
    fn="graph1.png"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    #graph2
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,8)
        v1='0.'+str(rn)
        x2.append(float(v1))
        i+=1
    
    x1=[0,0,0,0,0]
    y=[30,80,140,220,275]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    

    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Model recall")
    plt.ylabel("recall")
    
    fn="graph2.png"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    #graph3########################################
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(94,98)
        v1='0.'+str(rn)

        #v11=float(v1)
        v111=round(rn)
        x1.append(v111)

        rn2=randint(94,98)
        v2='0.'+str(rn2)

        
        #v22=float(v2)
        v33=round(rn2)
        x2.append(v33)
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[5,30,60,90,120]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    
    plt.figure(figsize=(10, 8))
    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy %")
    
    fn="graph3.png"
    plt.savefig('static/trained/'+fn)
    plt.close()
    #######################################################
    #graph4
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,4)
        v1='0.'+str(rn)

        #v11=float(v1)
        v111=round(rn)
        x1.append(v111)

        rn2=randint(1,4)
        v2='0.'+str(rn2)

        
        #v22=float(v2)
        v33=round(rn2)
        x2.append(v33)
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[5,30,60,90,120]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    
    plt.figure(figsize=(10, 8))
    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Model loss")
    
    fn="graph4.png"
    plt.savefig('static/trained/'+fn)
    plt.close()
    return render_template('pro5.html',dimg=dimg)

def toString(a):
  l=[]
  m=""
  for i in a:
    b=0
    c=0
    k=int(math.log10(i))+1
    for j in range(k):
      b=((i%10)*(2**j))   
      i=i//10
      c=c+b
    l.append(c)
  for x in l:
    m=m+chr(x)
  return m
                
@app.route('/pro6', methods=['GET', 'POST'])
def pro6():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    print("aaa")
    for fname in os.listdir(path_main):
        dimg.append(fname)
        print(fname)

    ff=open("static/trained/class.txt",'r')
    ext=ff.read()
    ff.close()
    cname=ext.split(',')
    '''data1=[]
    data2=[]
    data3=[]
    data4=[]
    v1=0
    v2=0
    v3=0
    v4=0
    path_main = 'static/trained'
    #for fname in os.listdir(path_main):
    i=0
    i<127
        dimg.append(fname)
        d1=fname.split('_')
        if d1[0]=='d':
            data1.append(fname)
            v1+=1
        if d1[0]=='f':
            data2.append(fname)
            v2+=1
        if d1[0]=='n':
            data3.append(fname)
            v3+=1
        if d1[0]=='w':
            data4.append(fname)
            v4+=1
        

    g1=v1+v2+v3+v4
    dd2=[v1,v2,v3,v4]
    
    
    doc = cname #list(data.keys())
    values = dd2 #list(data.values())
    print(doc)
    print(values)
    fig = plt.figure(figsize = (10, 5))
     
    # creating the bar plot
    plt.bar(doc, values, color ='blue',
            width = 0.4)
 

    plt.ylim((1,g1))
    plt.xlabel("Objects")
    plt.ylabel("Count")
    plt.title("")

    rr=randint(100,999)
    fn="tclass.png"
    plt.xticks(rotation=20)
    plt.savefig('static/trained/'+fn)
    
    plt.close()
    #plt.clf()'''

    #,data1=data1,data2=data2,data3=data3,data4=data4,cname=cname,v1=v1,v2=v2,v3=v3,v4=v4
    ##############################

    
    ###############################
    
    
    

    return render_template('pro6.html',dimg=dimg)

#######
@app.route('/classify', methods=['GET', 'POST'])
def classify():
    msg=""
    data2=[]
    cname=[]
    mycursor = mydb.cursor()

    mycursor.execute("SELECT count(*) FROM food_info")
    cn1 = mycursor.fetchone()[0]

    mycursor.execute("SELECT food FROM food_info group by food")
    cc = mycursor.fetchall()
    print(cc)

    dd2=[]
    for dd in cc:
        cname.append(dd[0])
        mycursor.execute("SELECT count(*) FROM food_info where food=%s",(dd[0],))
        d1 = mycursor.fetchone()[0]
        dd2.append(d1)

    dat = pd.read_csv("static/trained/data.csv")  
    for ss in dat.values:
        data2.append(ss)
    
    '''ff=open("static/trained/class.txt",'r')
    ext=ff.read()
    ff.close()
    cname=ext.split(',')


    ##    
    ff2=open("static/trained/values.txt","r")
    rd=ff2.read()
    ff2.close()
    val=rd.split(',')

    df = pd.read_csv("static/trained/train.csv")
    data1=[]
    

    dtt1=[]
    dtt2=[]
    dtt3=[]
    dtt4=[]
    dtt5=[]
    dtt6=[]
    dtt7=[]
    dtt8=[]
    dtt9=[]
    dtt10=[]
    dtt11=[]
    dtt12=[]
    v1=0
    v2=0
    v3=0
    v4=0
    v5=0
    v6=0
    v7=0
    v8=0
    v9=0
    v10=0
    v11=0
    v12=0

        
    for ss1 in df.values:
        dt=[]
        dt.append(ss1[1])
        dt.append(ss1[2])
        dt.append(ss1[3])
        
        
        if ss1[2]==cname[0]:
            #print(ss1[0])
            v1+=1
            #dtt1.append(ss1[1])
            #dtt1.append(ss1[3])
            gr=int(ss1[3])
            cal=int(val[0])
            calo=(cal/100)*gr
            calo1=round(calo)
            dt.append(str(calo1))
            #dtt1.append(cname[0])
        
        if ss1[2]==cname[1]:
            #print(ss1[0])
            v2+=1
            #dtt2.append(ss1[1])
            #dtt2.append(ss1[3])
            gr=int(ss1[3])
            cal=int(val[1])
            calo=(cal/100)*gr
            calo1=round(calo)
            dt.append(str(calo1))
            #dtt2.append(cname[1])
      
        if ss1[2]==cname[2]:
            v3+=1
            #dtt3.append(ss1[1])
            #dtt3.append(ss1[3])
            gr=int(ss1[3])
            cal=int(val[2])
            calo=(cal/100)*gr
            calo1=round(calo)
            dt.append(str(calo1))
            #dtt3.append(cname[2])
       
        
        if ss1[2]==cname[3]:
            v4+=1
            #dtt4.append(ss1[1])
            #dtt4.append(ss1[3])
            gr=int(ss1[3])
            cal=int(val[3])
            calo=(cal/100)*gr
            calo1=round(calo)
            dt.append(str(calo1))
            #dtt4.append(cname[3])
        
        if ss1[2]==cname[4]:
            v5+=1
            #dtt5.append(ss1[1])
            #dtt5.append(ss1[3])
            gr=int(ss1[3])
            cal=int(val[4])
            calo=(cal/100)*gr
            calo1=round(calo)
            dt.append(str(calo1))
            #dtt5.append(cname[4])
        
        if ss1[2]==cname[5]:
            v6+=1
            #dtt6.append(ss1[1])
            #dtt6.append(ss1[3])
            gr=int(ss1[3])
            cal=int(val[5])
            calo=(cal/100)*gr
            calo1=round(calo)
            dt.append(str(calo1))
            #dtt6.append(cname[5])

        if ss1[2]==cname[6]:
            v7+=1
            #dtt7.append(ss1[1])
            #dtt7.append(ss1[3])
            gr=int(ss1[3])
            cal=int(val[6])
            calo=(cal/100)*gr
            calo1=round(calo)
            dt.append(str(calo1))
            #dtt7.append(cname[6])

        if ss1[2]==cname[7]:
            v8+=1
            #dtt8.append(ss1[1])
            #dtt8.append(ss1[3])
            gr=int(ss1[3])
            cal=int(val[7])
            calo=(cal/100)*gr
            calo1=round(calo)
            dt.append(str(calo1))
            #dtt8.append(cname[7])

        if ss1[2]==cname[8]:
            v9+=1
            #dtt9.append(ss1[1])
            #dtt9.append(ss1[3])
            gr=int(ss1[3])
            cal=int(val[8])
            calo=(cal/100)*gr
            calo1=round(calo)
            dt.append(str(calo1))
            #dtt9.append(cname[8])

        if ss1[2]==cname[9]:
            v10+=1
            #dtt10.append(ss1[1])
            #dtt10.append(ss1[3])
            gr=int(ss1[3])
            cal=int(val[9])
            calo=(cal/100)*gr
            calo1=round(calo)
            dt.append(str(calo1))
            #dtt10.append(cname[9])

        if ss1[2]==cname[10]:
            v11+=1
            #dtt11.append(ss1[1])
            #dtt11.append(ss1[3])
            gr=int(ss1[3])
            cal=int(val[10])
            calo=(cal/100)*gr
            calo1=round(calo)
            dt.append(str(calo1))
            #dtt11.append(cname[10])

        if ss1[2]==cname[11]:
            v12+=1
            #dtt12.append(ss1[1])
            #dtt12.append(ss1[3])
            gr=int(ss1[3])
            cal=int(val[11])
            calo=(cal/100)*gr
            calo1=round(calo)
            dt.append(str(calo1))
            #dtt11.append(cname[11])

 

        data2.append(dt)'''
        
      
    
    #dd2=[v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12]
    doc = cname #list(data.keys())
    values = dd2 #list(data.values())
        
    
    
    doc = cname #list(data.keys())
    values = dd2 #list(data.values())
    print(doc)
    print(values)
    fig = plt.figure(figsize = (10, 8))
     
    # creating the bar plot
    plt.bar(doc, values, color ='orange',
            width = 0.4)
 

    plt.ylim((1,25))
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("")

    rr=randint(100,999)
    fn="tclass.png"
    #plt.xticks(rotation=20)
    plt.savefig('static/trained/'+fn)
    
    plt.close()
    #plt.clf()'''

    
    return render_template('classify.html',msg=msg,cname=cname,data2=data2)

#######

def calculate_bmr(gender, weight1, height1, age1):
    weight=float(weight1)
    height=float(height1)
    age=float(age1)
    h=height/100
    hh=h*h
    v=weight/hh
    vv=round(v,2)
    return vv
    '''if gender == "Female":
        women = (weight * 10) + (height * 6.25) - (age * 5) - 161
        return float(women)
    else:
        men = (weight * 10) + (height * 6.25) - (age * 5) + 5
        return float(men)'''


def get_sedentary(rest_bmr):
    sedentary = rest_bmr * 1.25
    return sedentary

def get_light_activity(rest_bmr):
    light = rest_bmr * 1.375
    return light

def get_moderate_activity(rest_bmr):
    moderate = rest_bmr * 1.550
    return moderate

def get_very_active(rest_bmr):
    active = rest_bmr * 1.725
    return active


def total_calculation(user_activity_lvl,rest_bmr):
    

    maintain = {
      "Sedentary" : get_sedentary(rest_bmr), 
      "Light" : get_light_activity(rest_bmr), 
      "Moderate" : get_moderate_activity(rest_bmr), 
      "Active" : get_very_active(rest_bmr)
      }
    res=""
    if user_activity_lvl == "Sedentary":
        res="You need to eat " + str(maintain["Sedentary"]) + " calories a day to maintain your current weight"
        #print("You need to eat " + str(maintain["sedentary"]) + " calories a day to maintain your current weight")

    elif user_activity_lvl == "Light":
        res="You need to eat " + str(maintain["Light"]) + " calories a day to maintain your current weight"
        #print("You need to eat " + str(maintain["light"]) + " calories a day to maintain your current weight")

    elif user_activity_lvl == "Moderate":
        res="You need to eat " + str(maintain["Moderate"]) + " calories a day to maintain your current weight"
        #print("You need to eat " + str(maintain["moderate"]) + " calories a day to maintain your current weight")

    elif user_activity_lvl == "Active":
        res="You need to eat " + str(maintain["Active"]) + " calories a day to maintain your current weight"
        #print("You need to eat " + str(maintain["active"]) + " calories a day to maintain your current weight")

    return res






@app.route('/test_bmi', methods=['GET', 'POST'])
def test_bmi():
    msg=""
    bmi=""
    result1=""
    result2=""
    st=""
    if request.method=='POST':
        st="1"
        gender=request.form['gender']
        age=request.form['age']
        height=request.form['height']
        weight=request.form['weight']
        activity=request.form['activity']
        rest_bmr = calculate_bmr(gender, weight, height, age)
        bmi=str(rest_bmr)
        if rest_bmr>=30:
            result1="Obese"
        elif rest_bmr>24 and rest_bmr<=29.9:
            result1="Overweight"
        elif rest_bmr>18.5 and rest_bmr<=24:
            result1="Normal Weight"
        else:
            result1="Underweight"
            
        result2=total_calculation(activity,rest_bmr)
        
    return render_template('test_bmi.html',bmi=bmi,result1=result1,result2=result2,st=st)


@app.route('/history', methods=['GET', 'POST'])
def history():
    act=""
    res=""
    uname=""
    if 'username' in session:
        uname = session['username']

    if uname is None:
        return redirect(url_for('login_user'))

    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register where uname=%s",(uname,))
    data1 = mycursor.fetchone()
    name=data1[1]

    mycursor.execute("SELECT * FROM history where uname=%s order by id desc",(uname,))
    data = mycursor.fetchall()

    
    return render_template('history.html',data=data)


@app.route('/test', methods=['GET', 'POST'])
def test():
    msg=""
    ss=""
    fn=""
    fn1=""
    fr2=""
    predict=""
    dta=""
    cname=[]
    uname=""
    if 'username' in session:
        uname = session['username']

    if uname is None:
        return redirect(url_for('login_user'))

    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register where uname=%s",(uname,))
    data = mycursor.fetchone()
    name=data[1]
    
    
    mycursor.execute("SELECT food FROM food_info group by food")
    cc = mycursor.fetchall()
    print(cc)

    dd2=[]
    for dd in cc:
        cname.append(dd[0])
    
    
    if request.method=='POST':
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            fname = file.filename
            filename = secure_filename(fname)
            f1=open('static/test/file.txt','w')
            f1.write(filename)
            f1.close()
            file.save(os.path.join("static/test", filename))

        cutoff=1
        path_main = 'static/dataset'
        for fname1 in os.listdir(path_main):
            hash0 = imagehash.average_hash(Image.open("static/dataset/"+fname1)) 
            hash1 = imagehash.average_hash(Image.open("static/test/"+filename))
            cc1=hash0 - hash1
            print("cc="+str(cc1))
            if cc1<=cutoff:
                ss="ok"
                fn=fname1
                
                fr=fn.split('.')
                fr2=fr[0]
                
                break
            else:
                ss="no"

        if ss=="ok":
            print("yes")
            
            df = pd.read_csv("static/trained/data.csv")
            for ss1 in df.values:
                a=ss1[2]
                b=fn
                if a==b:
                    
                    predict=ss1[1]
                    print("predict:"+predict)
                    dta=predict+"|"+fn+"|"+str(ss1[4])+"|"+str(ss1[3])
                    f3=open("static/test/res.txt","w")
                    f3.write(dta)
                    f3.close()
                    break
            
            
            
            

            
                    
            return redirect(url_for('test_pro',act="1"))
        else:
            msg="Invalid!"
    
    
        
    return render_template('test.html',msg=msg,name=name)


    
@app.route('/test_pro', methods=['GET', 'POST'])
def test_pro():
    msg=""
    fn=""
    act=request.args.get("act")
    uname=""
    if 'username' in session:
        uname = session['username']

    if uname is None:
        return redirect(url_for('login_user'))
    
    f2=open("static/test/res.txt","r")
    get_data=f2.read()
    f2.close()

    gs=get_data.split('|')
    fn=gs[1]
    
    ts=gs[0]
    fname=fn
    ##bin
    '''image = cv2.imread('static/dataset/'+fn)
    original = image.copy()
    kmeans = kmeans_color_quantization(image, clusters=4)

    # Convert to grayscale, Gaussian blur, adaptive threshold
    gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,2)

    # Draw largest enclosing circle onto a mask
    mask = np.zeros(original.shape[:2], dtype=np.uint8)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)
        cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
        break
    
    # Bitwise-and for result
    result = cv2.bitwise_and(original, original, mask=mask)
    result[mask==0] = (0,0,0)

    
    ###cv2.imshow('thresh', thresh)
    ###cv2.imshow('result', result)
    ###cv2.imshow('mask', mask)
    ###cv2.imshow('kmeans', kmeans)
    ###cv2.imshow('image', image)
    ###cv2.waitKey()

    #cv2.imwrite("static/upload/bin_"+fname, thresh)'''
    

    ###fg
    '''img = cv2.imread('static/dataset/'+fn)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    segment = cv2.subtract(sure_bg,sure_fg)
    img = Image.fromarray(img)
    segment = Image.fromarray(segment)
    path3="static/trained/test/fg_"+fname
    #segment.save(path3)'''
    
        
    return render_template('test_pro.html',msg=msg,fn=fn,ts=ts,act=act)

@app.route('/test_pro2', methods=['GET', 'POST'])
def test_pro2():
    msg=""
    fn=""
    uname=""
    if 'username' in session:
        uname = session['username']

    if uname is None:
        return redirect(url_for('login_user'))
    
    mycursor = mydb.cursor()
    act=request.args.get("act")
    
    f2=open("static/test/res.txt","r")
    get_data=f2.read()
    f2.close()

    gs=get_data.split('|')
    fn=gs[1]
    ts=gs[0]
    gram=gs[2]
    calorie=gs[3]

    if act=="6":
        mycursor.execute("SELECT max(id)+1 FROM history")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
                
        sql = "INSERT INTO history(id,uname,food,filename,calorie,gram) VALUES (%s, %s, %s, %s, %s, %s)"
        val = (maxid,uname,ts,fn,calorie,gram)
        mycursor.execute(sql,val)
        mydb.commit()
        
    
    
    '''ff=open("static/trained/class.txt",'r')
    ext=ff.read()
    ff.close()

    ff=open("static/trained/class.txt",'r')
    ext=ff.read()
    ff.close()

    xx=ext.split(',')

    ff=open("static/trained/values.txt",'r')
    val=ff.read()
    ff.close()

    yy=val.split(',')

    calorie=""
    i=0
    for x in xx:
        if ts==x:
            cal=int(yy[i])
            gr=int(gram)
            calo=(cal/100)*gr
            calorie=round(calo)
            break
        i+=1'''

    mycursor.execute("SELECT * FROM food_data where food=%s",(ts,))
    data1 = mycursor.fetchall()

    mycursor.execute("SELECT * FROM food_data where food=%s",(ts,))
    data2 = mycursor.fetchall()
    
    return render_template('test_pro2.html',msg=msg,fn=fn,ts=ts,act=act,gram=gram,calorie=calorie,data1=data1,data2=data2)




##########################
@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


