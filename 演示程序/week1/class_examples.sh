# ==== to produce the results of class examples

helpFunction()
{
   echo ""
   echo "Usage: $0 -m mode"
   echo "\t-m recog or entropy"
   exit 1 # Exit script after printing help
}

while getopts m: opt
do
    case "${opt}" in
        m) MODE=${OPTARG};;
        ?) helpFunction ;;
    esac
done

if [ $MODE = recog ]; then
	python recognition.py
	python recognition.py --mode test
elif [ $MODE = entropy ]; then
	python cal_entropy.py --n_bins 256
	python cal_entropy.py --mode single --n_bins 256 --im_path data/images/1647.jpg
	python cal_entropy.py --mode single --n_bins 256 --im_path data/images/1605.jpg
	python cal_entropy.py --mode single --n_bins 256 --im_path data/my_images/a.png --visualize
else
  helpFunction
fi
