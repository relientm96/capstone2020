import tempfile
import shutil
import errno
import os
import sys
'''
if not (os.path.exists(path_X) or os.path.exists(path_Y)):
		try:
			open(path_X, 'w').close()
			open(path_Y, 'w').close()
		except Exception as e:
			print("An error occured", e)
			sys.exit(-1)
'''
# exception handler  
def handler(func, path, exc_info):  
	print("Inside handler")  
	print(exc_info) 


# create the file within the temp directory;
def check_newfile(file_path):
	if not (os.path.exists(file_path)):
		try:
			open(file_path, 'w').close()
		except Exception as e:
			print("An error occured", e)
			sys.exit(-1)

if __name__ == '__main__':  
	# experiment 01
	try:
		# create temporary locations
		tmp_dir = tempfile.mkdtemp()
		x1_path = os.path.join(tmp_dir, "x1.txt")
		y1_path = os.path.join(tmp_dir, "y1.txt")
		
		# create the files within the temp directory;
		if not (os.path.exists(x1_path) or os.path.exists(y1_path)):
			try:
				open(x1_path, 'w').close()
				open(y1_path, 'w').close()
			except Exception as e:
				print("An error occured", e)
				sys.exit(-1)
		print("one: ", os.path.isdir(tmp_dir))
		print("two: ", os.path.exists(x1_path))
		print("two: ", os.path.exists(y1_path))
	# the created temporary paths are no longer needed;
	# clean up;
	finally:
		try:
			# delete directory
			shutil.rmtree(tmp_dir, onerror = handler)  
		except OSError as exc:
			 # ENOENT - no such file or directory
			if exc.errno != errno.ENOENT: 
				# re-raise exception
				raise  
	print("one: ", os.path.isdir(tmp_dir))
	print("two: ", os.path.exists(x1_path))


	# experiment 02
	with tempfile.TemporaryDirectory() as txt_path:
		print(txt_path)
		dummyfile =  os.path.join(txt_path, "dummy.txt")
		check_newfile(dummyfile)

		print("hello1: ", os.path.isdir(txt_path))
		print("hello2: ", os.path.exists(dummyfile))
	print("hello1: ", os.path.isdir(txt_path))
	print("hello2: ", os.path.exists(dummyfile))
				