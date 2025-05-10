set -x
python ../SageAttention/get_accuracy.py 
./build/sage_attention_test
python get_accuracy.py

echo "=================================================================================================================="
python ../SageAttention/get_accuracy.py cogvideox0
./build/sage_attention_test cogvideox0
python get_accuracy.py cogvideox0
echo "=================================================================================================================="
python ../SageAttention/get_accuracy.py cogvideox1
./build/sage_attention_test cogvideox1
python get_accuracy.py cogvideox1
echo "=================================================================================================================="
python ../SageAttention/get_accuracy.py cogvideo0
./build/sage_attention_test cogvideo0
python get_accuracy.py cogvideo0
echo "=================================================================================================================="
python ../SageAttention/get_accuracy.py cogvideo1
./build/sage_attention_test cogvideo1
python get_accuracy.py cogvideo1
echo "=================================================================================================================="
