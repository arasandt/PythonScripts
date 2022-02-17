# =============================================================================
# import binascii
# f1 = open("D:\\Arasan\\Misc\\GitHub\\VideoCapture\\Lenel_Video.asf", "rb")
# f2 = open("D:\\Arasan\\Misc\\GitHub\\VideoCapture\\Lenel_video_LNR_Format.lnr", "rb")
# try:
#     #byte1 = binascii.hexlify(f1.read(1))
#     byte2 = binascii.hexlify(f2.read(1))
#     arr1 = []
#     #while byte != "":
#     for i in range(4149 * 1024):
#         # Do stuff with byte.
#         #print('asf: {0}   lnr:{1}'.format(byte1, byte2))
#         #print('{0}'.format(byte1))
#         arr1.append(byte2)
#         #byte1 = f1.read(1)
#         #byte2 = f2.read(1)
#         #byte1 = binascii.hexlify(f1.read(1))
#         byte2 = binascii.hexlify(f2.read(1))
# finally:
#     print(arr1)
#     f1.close()
#     f2.close()
# =============================================================================
with open("D:\\Arasan\\Misc\\GitHub\\VideoCapture\\Lenel_video_LNR_Format.lnr", "rb") as binary_file:
    # Read the whole file at once
    data = binary_file.read(32)
    print(data)
    data = binary_file.read(16)
    print(data)
    
    #print(len(data)/1024)
    #print(data[0],data[1],data[2])
    #print(data)
# =============================================================================
# import cchardet
# 
# def convert_encoding(data, new_coding = 'UTF-8'):
#   encoding = cchardet.detect(data)['encoding']
#   print(encoding)
#   if new_coding.upper() != encoding.upper():
#     data = data.decode(encoding, data).encode(new_coding)
# 
#   return data
# message = "Hello"  # str
# data = message.encode('utf-8')
# convert_encoding(data)
# =============================================================================

#with open("D:\\Arasan\\Misc\\GitHub\\VideoCapture\\test.lnr", "wb") as binary_file:
#    # Write text or bytes to the file
#    #binary_file.write("Write text by encoding\n".encode('utf8'))
#    num_bytes_written = binary_file.write(data[:100000])
#    print("Wrote %d bytes." % num_bytes_written)