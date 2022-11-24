if [ -z $MODEL_PATH ]
then
    export MODEL_PATH="LogisticRegressionCV_model.pkl"
fi

if [[ ! -f $MODEL_PATH ]]
then
    wget https://drive.google.com/uc?export=download&id=16rgzlVKVWHQM8pWX-ik8gcctXhCpxTPa --output-document=$MODEL_PATH
    echo $MODEL_PATH
else
    echo "Model exists"
fi

uvicorn main:app --reload --host 0.0.0.0 --port 15000