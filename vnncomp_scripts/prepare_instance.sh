CONDA_HOME=~/conda
CONDA=$CONDA_HOME/bin/conda

DNNV_CONDA_HOME=~/.conda/envs/dnnv
DNNV_PY=$DNNV_CONDA_HOME/bin/python

# echo $DNNV_PYTHON
$DNNV_PY -c "import dnnv; print('DNNV version', dnnv.__version__)"
$DNNV_PY -c "import onnxsim; print('ONNXSIM version', onnxsim.__version__)"

CATEGORY=$2
ONNX_FILE=$3
VNNLIB_FILE=$4
RESULTS_FILE=$5
TIMEOUT=$6

echo "Simplifying for benchmark '$CATEGORY' with onnx file '$ONNX_FILE'"

TOOL_DIR=$(dirname $(dirname $(realpath $0)))
OUTPUT_DIR=$TOOL_DIR/tmp_simplified_model_output

echo TOOL_DIR = $TOOL_DIR
echo OUTPUT_DIR = $OUTPUT_DIR

if [ -d $OUTPUT_DIR ]; then
	rm -r $OUTPUT_DIR
fi

$DNNV_PY $TOOL_DIR/neuralsat-pt201/util/network/simplify_onnx.py $ONNX_FILE $OUTPUT_DIR/model-simplified
# if [ "${CATEGORY,,}" == "vggnet16" ] || [ "${CATEGORY,,}" == "cgan" ]; then
#     $DNNV_PY $TOOL_DIR/neuralsat-pt201/util/network/simplify_onnx.py $ONNX_FILE $OUTPUT_DIR/model-simplified
# fi
exit 0