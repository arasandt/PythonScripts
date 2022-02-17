import time
import win32pipe, win32file


def pipe_server():
    print("pipe server")
    count = 0
    
    message = ['Educated',
               'The Reckoning',
               'Cook Like a Pro',
               'There There',
               'The Soul of America',
               'Past Tense',
               'The Line Becomes a River',
               'Love and Ruin',
               'How to Change Your Mind',]
    
    pipe = win32pipe.CreateNamedPipe(
        r'\\.\pipe\Foo',
        win32pipe.PIPE_ACCESS_DUPLEX,
        win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
        1, 65536, 65536,
        0,
        None)
    try:
        print("waiting for client")
        win32pipe.ConnectNamedPipe(pipe, None)
        print("got client")

        while count < len(message):
            
            print(f"Sending.... {message[count]}")
            # convert to bytes
            some_data = str.encode(f"{message[count]}")
            win32file.WriteFile(pipe, some_data)
            time.sleep(1)
            count += 1

        print("finished now")
    finally:
        win32file.CloseHandle(pipe)




if __name__ == '__main__':
    pipe_server()
