def show_boundingbox(*args, **kwargs):
    """
    """
    
    
    import cv2, os
    from utils import Progress
    ex = 'D:\\Arasan\\Server 8 Channel 54\\1d5dfee3f75b0ef 2698323.h264.xlsx'
    vfile = 'D:\\Arasan\\Server 8 Channel 54\\1d5dfee3f75b0ef 2698323.h264.avi'
    from pathlib import Path
    p = Path(vfile)
    outputpath = os.path.join(p.parent, p.stem)
    Path(outputpath).mkdir(parents=True, exist_ok=True)
    
    const_videocompression = 'XVID'
    
    cap = cv2.VideoCapture(vfile)
    heightencoded = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    widthencoded = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    out = cv2.VideoWriter(vfile + '.out.avi', cv2.VideoWriter_fourcc(*const_videocompression), 10, (widthencoded,heightencoded))    
    import pandas as pd
    
    df = pd.read_excel(ex, sheet_name='Sheet2')
    
    count = 1
    
    with Progress('',int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
    
        while True:
            (ret, frame) = cap.read()
            if not ret:
                break
            
            count_framegroup = ((count - 1) // 10) + 1
            
            df_group = df.loc[(df['framegroup'] == count_framegroup)]
            
            cv2.putText(frame, str(len(df_group)), (40, 40), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 255), thickness=3, lineType=cv2.LINE_8)
            fn = 0
            
            for index, row in df_group.iterrows():
                coor = eval(row['box'])
                fn = row['framenumber']
                frame = cv2.rectangle(frame, (coor[0], coor[1]), (coor[0] + coor[2], coor[1] + coor[3]), (255,255,0), 1)
                        
            
            if fn == count :
                cv2.imwrite(os.path.join(outputpath ,'{0}_BBBBBB.png'.format(count)), frame)
            elif count == 0:
                pass
            elif count % 10 == 0:
                cv2.imwrite(os.path.join(outputpath , '{0}_A.png'.format(count)), frame)
            
            pbar.update(count)
            
            count += 1
            out.write(frame)
            

    out.release()
    
    
if __name__ == '__main__':
    """
    """
    
    show_boundingbox()
    
