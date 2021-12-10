from flask import  Response
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import io
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import base64
from PIL import Image
import tensorflow as tf
from skimage.io import imsave

FA = "https://use.fontawesome.com/releases/v5.15.2/css/all.css"
dash_app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP,FA])
app=dash_app.server

#Model sử dụng
face_classifier = cv2.CascadeClassifier(r'model/haarcascade_frontalface_default.xml')
model =tf.keras.models.load_model('model/mdfinal.h5')
class_labels = ['gian','So','vui','binh thuong','buon','ngac nhien']
labels = []
#Giao diên Camera
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        gray_img= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_img, 1.32, 5)
        for (x, y, w, h) in faces:
    # vẽ 1 đường bao quanh khuôn mặt
         cv2.rectangle(image,(x,y),(x+w,y+h),(50,205,50),2)
         cv2.rectangle(image, (x,y-40), (x+w,y), (50,205,50), -1)
         #Tách phần khuôn mặt vừa tìm được và resize về kích thước 48x48 để chuẩn bị đưa vào bộ mạng Neural Network
         roi_gray = gray_img[y:y+h,x:x+w]
         roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
         if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)  

            # Thực hiện dự đoán cảm xúc
            
            preds = model.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position = (x+20,y-10)  
            cv2.putText(image,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

         else:
             cv2.putText(image,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),3)
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


def gen(camera):
    while True:
        frame = camera.get_frame()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
                    
page_2_layout = html.Div([
    html.H1('NHẬN DIỆN CẢM XÚC GƯƠNG MẶT NGƯỜI BẰNG CAMERA',style={'textAlign':'center'}),
    html.Img(src ='/video_feed',style={"paddingLeft":380}),
])



#Màn hình nhận diện ảnh
layout = html.Div([
        html.Div(
            [html.Div(children='ĐỀ TÀI NHẬN DIỆN CẢM XÚC GƯƠNG MẶT NGƯỜI',
                     style={'marginLeft': '30px', 'width': '100%', 'marginTop': '100px','fontSize':'55px','fontFamily':'Georgia'}),
            html.Br(),
            # html.Div(children='BẰNG KERAS VÀ OPENCV',
            #          style={'marginLeft': '30px', 'width': '100%','color':'black','fontSize':'25px'})
            ]
            ,
            style={'height':550,'width':450,'backgroundColor':'#F0FFFF'}
        ),
        html.Div([
            html.Div(
                 html.Div(
                     dbc.Button(
                         dcc.Upload(children='TẢI ẢNH', id='upload-image',style={'width': '150px', 'height': '30px','textAlign':'center','paddingTop':5}),
                         color="danger", className="me-1",style={'marginLeft': '300px','marginTop': '20px',}
                     )
                 ),
                style={'height': 100, 'width': 750, 'backgroundColor': '#FAEBD7','marginBottom':15}
            ),
            html.Div(
                [
                    html.Div(
                        [
                        html.Div(children='ẢNH ĐẦU VÀO',
                                  style={'marginLeft': '120px', 'width': '100%', 'marginTop': '80px'}, id='kq'),
                        html.Br(),
                        html.Div(children=(
                                        html.Img(src='https://reactnativecode.com/wp-content/uploads/2018/02/Default_Image_Thumbnail.png',
                                                 id='output-image-upload', style={'height': '250px', 'width': '300px','marginLeft':25})),
                                        )
                        ],
                        style={'height': 435, 'width': 350}
                    ),
                    html.Div(
                        [
                        html.Div(children='ẢNH ĐƯA VÀO DỰ ĐOÁN',
                                  style={'marginLeft': '40px', 'width': '100%', 'marginTop': '80px','marginBottom': '80px'}),
                        html.Br(),
                        html.Div(children=(
                                        html.Img(src='https://reactnativecode.com/wp-content/uploads/2018/02/Default_Image_Thumbnail.png',
                                                  style={'height': '100px', 'width': '100px','marginLeft':85},id='output-process-image-upload')),
                                        )
                        ],
                        style={'height': 435, 'width': 250}
                    ),
                    html.Div(
                        [html.Div(children='DỰ ĐOÁN',
                                  style={'marginLeft': '40px', 'width': '100%', 'marginTop': '80px','marginBottom': '80px'}),
                        html.Br()
                            ,
                        html.Div(
                                style={'width': '80%', 'height': '50px', 'lineHeight': '60px', 'textAlign': 'center',
                                       'background': '#FFFAF0', 'borderWidth': '1px', 'borderStyle': 'double',
                                       'marginLeft': '15px','marginTop': '5px','color':'black','fontSize':'20px'},id = 'pred')
                        ],
                        style={'height': 435, 'width': 150}
                    )
                ],
                style={'height': 435, 'width': 750, 'backgroundColor': '#FFEBCD','display':'flex','flex-direction':'row'}
            )
        ],
            style={'height':550,'width':750,'marginLeft':30}
        )
],style={'display':'flex','flex-direction':'row','height':'80%','width':'100%','paddingLeft':60})

def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(imgdata))
    return image
def detect_image(image):
    face_classifier = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    return roi_gray
def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    image = stringToImage(data)
    mang = np.array(image)
    img = detect_image(mang)
    mang1 = np.array(img)
    imsave(name,mang1)   

@dash_app.callback(
    Output('output-image-upload','src'),
    [Input("upload-image", "filename"),
     Input("upload-image", "contents")],
)
def upload(filename,contents):
    if contents is not None and filename is not None:
        return contents
    else:
        raise dash.exceptions.PreventUpdate()
@dash_app.callback(
    [Output('output-process-image-upload','src'),
     Output('pred','children')],
    [Input("upload-image", "filename"),
     Input("upload-image", "contents")],
)
def process_image_upload(filename,contents):
    if contents is None and filename is None:
        raise dash.exceptions.PreventUpdate()
    index_word = {0:'Tức giận',1:'Sợ hãi',2:'Vui vẻ',3:'Bình thường',4:'Buồn bã', 5:'Bất ngờ'}
    data = contents.encode("utf8").split(b";base64,")[1]
    image = stringToImage(data)
    mang = np.array(image)
    img = detect_image(mang)
    img1 = img.reshape(1,48,48,1)
    pred = np.argmax(model.predict(img1))
    kq = index_word[pred]
    save_file(filename,contents)
    image = filename
    encoded = base64.b64encode(open(image, 'rb').read())
    encoded1 = 'data:image/png;base64,{}'.format(encoded.decode())
    return encoded1,kq


#Màn Hình Main
dash_app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        html.Div(
            dbc.Button(dcc.Link('Recoginze_Camera', href='/apps/camera'),outline=True, color="success", className="me-1"),
            style={'height':'50px','width':'50%','paddingLeft':500}
        ),
        html.Div(
            dbc.Button(dcc.Link('Recoginze_Image', href='/apps/recoginze_image'), outline=True, color="warning", className="me-1"),
            style={'height':'50px','width':'50%'}
        ),
        # dbc.Button(dcc.Link('Recoginze_Image', href='/apps/recoginze_image'), color="dark", className="me-1",style={'height':'50px','width':'50%'}),
    ], style={'display':'flex','flex-direction':'row','height':50,'width':'100%'},className="row"),
    html.Div(id='page-content', children=[])
])

@dash_app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/recoginze_image':
        return layout
    if pathname == '/apps/camera':
        return page_2_layout