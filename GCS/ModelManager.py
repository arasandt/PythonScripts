#!/usr/bin/python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import cv2, os
import json, ffmpeg
import ImageProcess, Model
import Utils, sys

import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from dateutil import tz
from Utils import Progress, range_group
from subprocess import Popen, PIPE

class ManagerObject:
    """
    """
    
    OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"boosting": cv2.TrackerBoosting_create,
		"mil": cv2.TrackerMIL_create,
		"tld": cv2.TrackerTLD_create,
		"medianflow": cv2.TrackerMedianFlow_create,
		"mosse": cv2.TrackerMOSSE_create
	 }
 
  
    def __init__(self, *args, **kwargs):
        """
        """
        
        classlocation = os.path.dirname(os.path.abspath(sys.modules[self.__class__.__module__].__file__))
        self.debug = args[5]
        self.settings = Utils.load_constants(self, classlocation, self.debug)
        self.const_videoextdef   = '.{0}.'.format(self.const_videocompressiondef)        
        self.const_ffmpegextdef  = '.{0}.'.format(self.const_ffmpegcompressiondef)        
        self.const_ffmpegextpipe = '.{0}.'.format(self.const_ffmpegcompressionpipe)        
        
        self.process = args[1]
        self.processstarttime = datetime.utcnow()
        self.filenamepath = args[0]
        self.filename = os.path.basename(self.filenamepath)
        self.videometafilename = self.filenamepath + self.const_metadataext
        self.framemetafilename = self.filenamepath + self.const_metadatacompressionext
        self.outputfilename = None
        self.rerun = kwargs.get('rerun', False)
        self.modelfolder = os.path.join(args[2].replace("'",""),"")
        self.framenum = args[3] if any(args[3]) else False
        self.const_tracker = self.const_tracker.lower()

        if self.rerun:
            pass #self.const_framerank = self.const_framerank + 1

    
    def load_metadata(self, *args, **kwargs):
        """
        """
        
        self.frame_meta = pd.read_parquet(self.framemetafilename)
        with open(self.videometafilename) as json_file:
            self.video_meta = json.load(json_file)

        self.video_meta[self.filename][self.process] = {}    

        self.filenamepathconverted = self.video_meta[self.filename]['fileattributes']['filenamepathconverted']

    
    def save_attributes(self, *args, **kwargs):
        """
        Save attributes to json file
        """
        
        self.processendtime = datetime.utcnow()
        
        self.video_meta[self.filename][self.process].update({
                'starttime'         : self.processstarttime.strftime('%m/%d/%Y %I:%M:%S %p'),
                'endtime'           : self.processendtime.strftime('%m/%d/%Y %I:%M:%S %p'),
                'duration'          : (self.processendtime - self.processstarttime).total_seconds(),
                'runsettings'       : self.settings,
                })               
        
        totalprocessduration = 0
        dict_keys = [i for i in self.video_meta[self.filename].keys() if i != 'fileattributes']
        for i in dict_keys:
            totalprocessduration += self.video_meta[self.filename][i].get('duration',0)
        
        self.video_meta[self.filename]['fileattributes'].update({'totalprocessduration': totalprocessduration})      
        
        with open(self.filenamepath + self.const_metadataext, 'w') as fp:
            json.dump(self.video_meta, fp)

        self.frame_meta.to_csv(self.filenamepath + '.csv',header=True, sep=',', index=False)
        self.frame_meta.to_parquet(self.framemetafilename, compression=self.const_metadatacompression)    


    def create_db_extracts(self, *args, **kwargs):
        """
        """
        
        filetable = pd.DataFrame.from_dict(self.video_meta[self.filename]['fileattributes'], orient='index').T
        filetable.to_csv(self.filenamepath + '_file.csv',header=True, sep=',', index=False)    
        
        framestable = self.frame_meta.drop(['model'], axis=1, inplace=False, errors='ignore')
        framestable.insert (0, "filename", self.filename)
        def get_count(x):
            try:
                p = eval(x)
                return len(p)
            except:
                return x
        
        framestable['personcount'] = framestable['persondetails'].apply(get_count)
        framestable['trackercount'] = framestable['tracker'].apply(get_count)
        framestable.drop(['persondetails','tracker'], axis=1, inplace=True, errors='ignore')
        framestable.to_csv(self.filenamepath + '_frames.csv',header=True, sep=',', index=False)    

        framedetailstable = self.frame_meta[['framenumber','model','persondetails','tracker','personfound']].reset_index()
        framedetailstable = framedetailstable.loc[(self.frame_meta['personfound'] == 1)]
        framedetailstable.drop(['personfound'], axis=1, inplace=True, errors='ignore')
        
        temp_dict = {}
        idx = 0 
        for index, row in framedetailstable.iterrows(): 
            if row['persondetails'] == "":
                pass
            else:
                objects = eval(row['persondetails'])
                for cnt, obj in enumerate(objects):
                    bb, confi = obj
                    temp_dict[idx] = {'filename': self.filename,
                                      'framenumber': row['framenumber'],
                                      'object': 'person',
                                      'model': row['model'],
                                      'sequence': cnt,
                                      'box': bb,
                                      'confidence': confi,
                                     }
                    idx += 1
            if row['tracker'] == "":
                pass
            else:
                objects = eval(row['tracker'])
                for cnt, obj in enumerate(objects):
                    bb, confi = obj
                    temp_dict[idx] = {'filename': self.filename,
                                      'framenumber': row['framenumber'],
                                      'object': 'tracker',
                                      'model': 'tracker',
                                      'sequence': cnt,
                                      'box': bb,
                                      'confidence': confi,
                                     }
                    idx += 1            
                    
                    
        framedetailstable = pd.DataFrame.from_dict(temp_dict, 'index')
        framedetailstable.to_csv(self.filenamepath + '_framedetails.csv',header=True, sep=',', index=False)    
                    
    
    def prepare_data(self, *args, **kwargs):
        """
        """
        
        # get fps
        fps = self.video_meta[self.filename]['fileattributes']['framerateencoded']
        
        # clean columns
        self.frame_meta.drop(['fgmeangroupsum','fgmeangrouprank','framegroup'], axis=1, inplace=True, errors='ignore')
        
        #check if background subtraction was applied
        motionapplied = len(self.frame_meta.loc[(self.frame_meta['fgmean'] != 1)])

        # group frame into 1 sec batches
        self.frame_meta['framegroup'] = 1 + (self.frame_meta['framenumber'] - 1) // fps
        
        # apply fgmean sum to all batch records and get all motion batches
        df = self.frame_meta.groupby(['framegroup'], as_index=False)['fgmean'].mean()
        df.rename(columns = {'fgmean':'fgmeangroupsum'}, inplace = True) 
        
        if motionapplied:
            dfm = df[df['fgmeangroupsum'] > self.const_fgmeanthreshold].copy()
            frameswithmotion = len(dfm) * fps
        else:
            frameswithmotion = len(df) * fps
        
        if self.const_applymotioncapture:
            df = df[df['fgmeangroupsum'] > self.const_fgmeanthreshold]

        # merge metadata with motion data
        self.frame_meta = pd.merge(self.frame_meta, df, on='framegroup',how='left').fillna(0)

        # rank the fgmean values
        self.frame_meta['temprank'] = self.frame_meta.groupby(['framegroup'], as_index=False)['fgmean'].rank("min", ascending=False)
        self.frame_meta['fgmeangrouprank'] = self.frame_meta.sort_values(['temprank']).groupby(['framegroup']).cumcount() + 1
        self.frame_meta.drop(['temprank'], axis=1, inplace=True, errors='ignore')
        
        motionpercentage = round((frameswithmotion * 100.0 / self.video_meta[self.filename]['fileattributes']['totalframes']),2)
        motionpercentage = min(motionpercentage, 100.00)
        self.video_meta[self.filename]['fileattributes'].update({'motionpercentage': motionpercentage})        
        

    def validate_model(self, *args, **kwargs):
        """
        """

        if self.modelfolder is None or not os.path.isdir(self.modelfolder):
            raise Exception('Model folder {0} is incorrect'.format(self.modelfolder))


    def select_frames_for_model(self, *args, **kwargs):
        """
        """
        
        if self.filenamepathconverted is None or not os.path.exists(self.filenamepathconverted):
            raise Exception('Converted input video file {0} not found'.format(self.filenamepathconverted))
        
        if self.framenum:
            # select only the frame which was passed as input in framenum field
            roi_frames = self.frame_meta.loc[(self.frame_meta['framenumber'].isin(list(self.framenum)))]
        else:
            # select only frames which have motion and matches with framerank.
            roi_frames = self.frame_meta.loc[(self.frame_meta['fgmeangroupsum'] != 0.0) & (self.frame_meta['fgmeangrouprank'] == self.const_framerank)]
            
            if self.rerun:
                roi_frames = roi_frames.loc[(roi_frames['personfound'] == 0.0)]

    
        #zip the columns
        self.model_frames = list(zip(list(roi_frames['framenumber']),list(roi_frames['framegroup'])))
       
        print('Selecting {0} image(s)'.format(len(self.model_frames)))
        
        self.model_frames = ImageProcess.get_images(self.filenamepathconverted, 
                                                    self.model_frames)
    
    def apply_model_on_singled_frame(self, *args, **kwargs):
        """
        """
        
        activity = 'Applying Model ({0})'.format(self.modelfolder)
        
        # Initialize the model
        persondet = Model.ModelObject(self.filenamepath, self.modelfolder, debug=self.debug) 
        if self.settings is not None:
            self.settings.update(persondet.settings)
        
        model_success = 0
        
        with Progress(activity, len(self.model_frames)) as pbar:
            
            for tup in range(len(self.model_frames)):
                frame, group, image = self.model_frames[tup]
                persondet.image = image
                persondet.framenumber = frame 
                result = persondet.detect()
                   
                self.model_frames[tup] = (frame, group, image, result)                

                if result != -1:
                    model_success += 1
                    
                pbar.update(tup + 1)

        print('Model successful on {0} image(s)'.format(model_success))

        for tup in range(len(self.model_frames)):
            frame, group, image, result = self.model_frames[tup]

            if result != -1:
                image = ImageProcess.apply_model_result(frame, group, image, result, showmessage=True)
            else:
                print('Frame #{0} => No result'.format(frame))

            outname = self.filenamepath + '#' + str(frame) +'.jpg' 
            cv2.imwrite(outname,image)
        
        print('All frame(s) saved as "{0}#<framenumber>.jpg"'.format(self.filename))
            
        
    def apply_model_on_frames(self, *args, **kwargs):
        """
        """
        
        activity = 'Applying Model ({0})'.format(self.modelfolder)
        
        if not self.rerun:
            self.frame_meta.drop(['personfound','model','persondetails'], axis=1, inplace=True, errors='ignore')
            self.frame_meta['personfound'] = 0.0
            self.frame_meta['model'] = ''
            self.frame_meta['persondetails'] = ''        
        
        # Initialize the model
        persondet = Model.ModelObject(self.filenamepath, self.modelfolder, debug=self.debug)
        if self.settings is not None:
            self.settings.update(persondet.settings)
        
        model_success = 0
        
        with Progress(activity, len(self.model_frames)) as pbar:
        
            for tup in range(len(self.model_frames)):
                frame, group, image = self.model_frames[tup]
                persondet.image = image
                persondet.framenumber = frame 
                result = persondet.detect()
                   
                self.model_frames[tup] = (frame, group, result)
                self.frame_meta.loc[self.frame_meta['framenumber'] == frame , ['model']] = self.modelfolder 

                if result != -1:
                    self.frame_meta.loc[self.frame_meta['framegroup'] == group, ['personfound']] = 1
                    self.frame_meta.loc[self.frame_meta['framenumber'] == frame , ['persondetails']] = str(result)  
                    model_success += 1
                
                pbar.update(tup + 1)
             
        print('Model successful on {0} image(s)'.format(model_success))
        
        self.video_meta[self.filename][self.process].update({
                'model': self.modelfolder,
                'selectedframes' : len(self.model_frames),
                'validframes'    : model_success,
                })          
    
    
    def perform_tracking(self, *args, **kwargs):
        """
        """

        activity = 'Applying OpenCV Tracker ({0})'.format(self.const_tracker)
        
        if self.filenamepathconverted is None or not os.path.exists(self.filenamepathconverted):
            raise Exception('Converted input video file {0} not found'.format(self.filenamepathconverted))

        def get_image_for_frame_number(selected_frames, frame_number):
            for frame in selected_frames:
                f, g, i = frame
                if f == frame_number:
                    return i
        
        self.frame_meta['tracker'] = ""
        
        df = self.frame_meta.loc[(self.frame_meta['personfound'] == 1)][['framenumber','framegroup','persondetails','fgmeangrouprank']]

        print('Selecting {0} image(s)'.format(len(df)))
        
        selected_frames = ImageProcess.get_images(self.filenamepathconverted, 
                                                  list(zip(df['framenumber'],df['framegroup'])))
        # make sure frame selection is done before reversing. else the same frame will be resturned
        
        tracklets = []
        tracklets_dict = {}
        tup = 0
        
        with Progress(activity, len(df)) as pbar:
            
            for index, row in df.iterrows():
                img = get_image_for_frame_number(selected_frames, row['framenumber'])
                if row['persondetails'] == "":
                    if tracklets:
                        tracklets_newbb = []
                        for track in tracklets:
                            (success, box) = track.update(img)
                            if success:
                                #(x, y, w, h) = [int(v) for v in box]
                                tracklets_newbb.append(([int(v) for v in box], 0))
                        tracklets_dict[index] = {'tracker': str(tracklets_newbb)}
                else:
                    tracklets = [] # reset trackers when person was found using person detection
                    persons = eval(row['persondetails'])
                    for det in persons:
                        personbb, confidence = det
                        tracklet = self.OPENCV_OBJECT_TRACKERS[self.const_tracker.lower()]()
                        tracklet.init(img, tuple(personbb))
                        tracklets.append(tracklet)
                
                tup += 1
                pbar.update(tup)
        
        df_tracker = pd.DataFrame.from_dict(tracklets_dict, 'index')
        self.frame_meta.update(df_tracker)
            
    
    def write_model_frames_with_result(self, *args, **kwargs):
        """
        """

        if self.filenamepathconverted is None or not os.path.exists(self.filenamepathconverted):
            raise Exception('Converted input video file {0} not found'.format(self.filenamepathconverted))
        
        fps = self.video_meta[self.filename]['fileattributes']['framerateencoded']
        width = self.video_meta[self.filename]['fileattributes']['widthencoded']
        height = self.video_meta[self.filename]['fileattributes']['heightencoded']
        
        df = self.frame_meta.loc[(self.frame_meta['writeflag'] != 0.0)][['framenumber','framegroup','persondetails','tracker']]
        
        if len(df) :
            #merge tracker information with persondetails
            df['persondetails'] = df.apply(lambda x: x['persondetails'] if x['persondetails'] != "" else x['tracker'], axis=1)
            
            df = df.replace(r'^\s*$', np.nan, regex=True)
            df['persondetails'].fillna(0, inplace=True)
            
            inputfilename = self.filenamepath + self.const_videoextdef + 'result.' + self.const_videocontainerdef 
            out = cv2.VideoWriter(inputfilename ,cv2.VideoWriter_fourcc(*self.const_videocompressiondef.upper()),fps, (width,height))    
            
            ImageProcess.get_images(self.filenamepathconverted, 
                                    list(zip(df['framenumber'],df['framegroup'])), 
                                    outwrite=out, 
                                    applyresult=list(df['persondetails']))
            
            out.release()
            
            outputfilename = self.filenamepath + self.const_ffmpegextdef + 'result.' + self.const_ffmegcontainerdef
            
            out, _ = (
                        ffmpeg
                        .input(inputfilename)
                        .output(outputfilename, crf=self.const_ffmpegcrfdef, vcodec=self.const_ffmpegcompressiondef)
                        .overwrite_output()
                        .run()
                     )
            
            # remove the temp video file    
            os.remove(inputfilename)
            
        
    

    def write_model_frames(self, *args, **kwargs):
        """
        """

        if self.filenamepathconverted is None or not os.path.exists(self.filenamepathconverted):
            raise Exception('Converted input video file {0} not found'.format(self.filenamepathconverted))
        
        fps = self.video_meta[self.filename]['fileattributes']['framerateencoded']
        width = self.video_meta[self.filename]['fileattributes']['widthencoded']
        height = self.video_meta[self.filename]['fileattributes']['heightencoded']
        
        df = self.frame_meta.loc[(self.frame_meta['writeflag'] != 0.0)][['framenumber','framegroup']]
        self.outputfilename = ""
        
        if len(df):
            if self.const_ffmpegpipe:
    
                self.outputfilename = self.filenamepath + self.const_ffmpegextpipe + self.const_ffmegcontainerpipe
        
                parms = map(str,['ffmpeg', '-y', '-f', 'image2pipe', 
                                 '-vcodec', self.const_ffmpegcodecpipe, '-r', fps, '-i', '-', 
                                 '-vcodec', self.const_ffmpegcompressionpipe, '-crf', 
                                 self.const_ffmpegcrfpipe,'-r', fps, self.outputfilename])
        
                p = Popen(parms, stdin=PIPE)
                
                ImageProcess.get_images(self.filenamepathconverted, 
                                        list(zip(df['framenumber'],df['framegroup'])), 
                                        outwrite=p.stdin, 
                                        fmt=self.const_ffmpegfmtpipe)
                
                p.stdin.close()
                p.wait()
                
            else:
                inputfilename = self.filenamepath + self.const_videoextdef + self.const_videocontainerdef
                out = cv2.VideoWriter(inputfilename ,cv2.VideoWriter_fourcc(*self.const_videocompressiondef.upper()),fps, (width,height))    
        
                ImageProcess.get_images(self.filenamepathconverted, 
                                        list(zip(df['framenumber'],df['framegroup'])), 
                                        outwrite=out)
                
                out.release()
                
                self.outputfilename = self.filenamepath + self.const_ffmpegextdef + self.const_ffmegcontainerdef
                
                out, _ = (
                            ffmpeg
                            .input(inputfilename)
                            .output(self.outputfilename, crf=self.const_ffmpegcrfdef, vcodec=self.const_ffmpegcompressiondef)
                            .overwrite_output()
                            .run()
                         )
                
                # remove the temp video file    
                os.remove(inputfilename)

        self.video_meta[self.filename][self.process].update({
                'filename': self.outputfilename,
                'totalframes': len(df),
                })          
        
        self.write_model_frames_with_result(*args, **kwargs)
        

    def post_process(self, *args, **kwargs):
        """
        """
        
        fps = self.video_meta[self.filename]['fileattributes']['framerateencoded']
        frames = self.video_meta[self.filename]['fileattributes']['totalframes']
        
        df = self.frame_meta.loc[(self.frame_meta['personfound'] != 0.0)][['framenumber']]
        
        self.frame_meta['writeflag'] = 0
        self.frame_meta['newframenumber'] = 0

        if len(df):
            grp = list(range_group(df['framenumber'].tolist()))
            
            for cnt in range(len(grp)):
                sframe, eframe = grp[cnt]
                newsframe = max(sframe - (fps * self.const_bufferseconds), 1) 
                neweframe = min(eframe + (fps * self.const_bufferseconds), frames)
                grp[cnt] = (newsframe, neweframe)
                self.frame_meta.loc[newsframe - 1: sframe - 1, 'writeflag'] = 2
                self.frame_meta.loc[sframe - 1: eframe, 'writeflag'] = 1
                self.frame_meta.loc[eframe: neweframe - 1, 'writeflag'] = 2
            
            df = self.frame_meta.loc[(self.frame_meta['writeflag'] != 0.0)].copy()
            df['newframenumber'] = 0
            df['newframenumber'] = df.groupby('newframenumber').cumcount() + 1
            self.frame_meta.update(df)                                  
            

    def validate_inp_outp(self, *args, **kwargs):
        """
        """
        
        df = self.frame_meta.loc[(self.frame_meta['writeflag'] != 0.0)][['framenumber','framegroup']]
        maxnnewframe = self.frame_meta['newframenumber'].max()
        
        if len(df):
            opname = self.video_meta[self.filename]['output']['filename']        
            cap = cv2.VideoCapture(opname)
            attributesop = {
    			"height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    			"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    			"fps": int(cap.get(cv2.CAP_PROP_FPS)),
    			"frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
              "newframes": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    		}
            cap.release()
            
            attributesin = {
    			"height": self.video_meta[self.filename]['fileattributes']['heightencoded'],
    			"width": self.video_meta[self.filename]['fileattributes']['widthencoded'],
    			"fps": self.video_meta[self.filename]['fileattributes']['framerateencoded'],
    			"frames": self.video_meta[self.filename]['output']['totalframes'],
              "newframes": maxnnewframe,
    		}        
            
            
            assert attributesin == attributesop, 'Validation failure between input and output\nInput  {0}\nOutput {1}'.format(attributesin,attributesop)
            
            ofile = self.video_meta[self.filename]['output']['filename'] 
            os.remove(ofile)
            os.remove(self.filenamepathconverted)
        
        
    def create_subtitles(self, *args, **kwargs):
        """
        """
        
        starttime = datetime.strptime('00:00:00', "%H:%M:%S")  
        secsCount = 0 
        timeformat = '%m/%d/%Y %I:%M:%S %p %Z'
        from_zone, to_zone = tz.gettz('UTC'), tz.gettz(self.const_subtitletimezone)
        
        ofile = self.video_meta[self.filename]['output']['filename'] 
        
        if ofile == "":
            pass
        else:            
            srtfilename = ofile + self.const_srtext
            vstarttime = self.video_meta[self.filename]['fileattributes']['starttimeencoded']
            vstarttime = datetime.strptime(vstarttime, "%m/%d/%Y %I:%M:%S %p")
            vstarttime = vstarttime.replace(tzinfo=from_zone)
            vstarttime = vstarttime.astimezone(to_zone)
            
            fps = self.video_meta[self.filename]['fileattributes']['framerateencoded']       
            
            allframes = self.frame_meta[self.frame_meta['writeflag'] != 0.0]['framenumber'].tolist()
    
            with open(srtfilename, "w+") as srtout:
                
                for i in range(0, len(allframes), fps):
                    srtout.write('{0}\n'.format(str(secsCount)))
                    
                    srtout.write('{0},000-->{0},999\n'.format(starttime.strftime('%H:%M:%S')))
                    starttime += timedelta(seconds = 1) 
                    
                    nexttime = vstarttime + timedelta(seconds = (allframes[i] // fps))
                    srtout.write('{0}\n'.format(nexttime.strftime(timeformat)))
    
                    srtout.write('\n')
                    secsCount += 1
            