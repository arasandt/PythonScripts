import os 
  
def communication(child_writes): 
    # file descriptors r, w for reading and writing 
    r, w = os.pipe() 
      
    #Creating child process using fork 
    processid = os.fork() 
    if processid: 
        # This is the parent process 
        # Closes file descriptor w 
        os.close(w) 
        r = os.fdopen(r) 
        print ("Parent reading") 
        str = r.read() 
        print( "Parent reads =", str) 
    else: 
        # This is the child process 
        os.close(r) 
        w = os.fdopen(w, 'w') 
        print ("Child writing") 
        w.write(child_writes) 
        print("Child writes = ",child_writes) 
        w.close() 
          
# Driver code         
child_writes = "Hello geeks"
communication(child_writes) 
  
# Contributed by "Sharad_Bhardwaj". 