TOOL_DIR=$(dirname $(pwd))
VENV_PATH=$TOOL_DIR/venv/bin/activate
echo $VENV_PATH

export NEURALSAT_ASSERT="true"
source $VENV_PATH
