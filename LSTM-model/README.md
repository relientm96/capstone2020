## LSTM python

* i have successfully run the code outside of the jupyter notebook;
* again, please ensure you have the right environment; i have attached "environment02.yml" for convenience;
* with LSTM.py, now we could:
  * save the model during training after every certain number of iterations as:
    * a backup;
    * we could monitor what is the performance; it could be the case that for certain subset of data, the performance is better; this shall be studied once we have a working model!
  * save the model after the training;
  * restoring the (final) model; this means that we dont have to retrain;
    * tested for the "X_val.txt" in the dataset;
 
## steps/remarks
1. make sure you reference to your path for the dataset;
```
[code line 166] DATASET_PATH = "C:\\Users\\yongw4\\Desktop\\yick\\lstm-tutorial\\dataset\\"
```
2. change the following to your own need; if the model has been trained, then 
```
[code line 279] RETRAIN = False
[otherwise]     RETRAIN = true
```
3. note that the models (during and after training) will be saved to the current directory; this will be a mess; sorry;
  * apparently, there are some issues with tensorflow (ver < 2) in handling relative path;
```
    if (TRAINING_STEPS % (100*batch_size)) == 0:
        # [issue] right now save at the current directory; failed at relative path; 
		save_path = saver.save(sess, "iter_model", global_step = step)
		print("Model saved in file: %s" % save_path)
	
```
```
    save_path = saver.save(sess, "saved_final_model")
	print("Model saved in file: %s" % save_path)
```
3.1 by above, note that for each saving, we will have three files generated, for example:
```
iter_models.data-xxxx-of-xxxx
iter_models.index
iter_models.meta_
```

## to-do
* perform further testing
*  test the prediction
* use tf.SavedModel() API for deploymeny;
    * technically, by above, we already able to run the prediction online if run in the active tensorflow session; but this is not elegant and not suitable for deployment;##
* to abstract out the functions
* to share the materials/online sources for these stufffs!
