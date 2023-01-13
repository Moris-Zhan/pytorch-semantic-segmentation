#----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#----------------------------------------------------#
import time
import os
import cv2
import numpy as np
from PIL import Image
import importlib
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Attribute Learner')
    parser.add_argument('--config', type=str, default="configs.pspnet_base" 
                        ,help = 'Path to config .opt file. Leave blank if loading from opts.py')
    parser.add_argument("--mode", type=str, default="video" , help="predict or video")  
    parser.add_argument("--video_fps", type=float, default=25.0, help="video_fps")  
    parser.add_argument("--test_interval", type=int, default=100, help="test_interval") 

    parser.add_argument("--video_path", type=str, 
                                        default="/home/leyan/DataSet/LANEdevkit/Drive-View-Noon-Driving-Taipei-Taiwan.mp4", 
                                        )  
    parser.add_argument("--video_save_path", type=str, 
                                        default="pred_out/coco.mp4", 
                                        ) 
    parser.add_argument("--dir_origin_path", type=str, 
                                        default="img/", 
                                        )  
    parser.add_argument("--dir_save_path", type=str, 
                                        default="img_out/", 
                                        )  

    conf = parser.parse_args() 
    opt = importlib.import_module(conf.config).get_opts(Train=False)
    for key, value in vars(conf).items():     
        setattr(opt, key, value)
    
    d=vars(opt)
    model = opt.Model_Pred(num_classes=opt.num_classes)   
    mode = opt.mode
    #----------------------------------------------------------------------------------------------------------#
    video_path      = opt.video_path
    video_save_path = opt.video_save_path
    video_fps       = opt.video_fps
    test_interval = opt.test_interval
    #----------------------------------------------------------------------------------------------------------#
    dir_origin_path = opt.dir_origin_path
    dir_save_path   = opt.dir_save_path
    fps_image_path  = "img/fps.jpg"
    #-------------------------------------------------------------------------#

    if mode == "video":
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            size = (1280, 720)
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while(True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break

            frame = cv2.resize(frame, (1280, 720), interpolation = cv2.INTER_AREA)
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(model.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open('img/street.jpg')
        tact_time = model.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
        
    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = model.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
                
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
