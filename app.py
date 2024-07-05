from flask import Flask, request, render_template
import pandas as pd
from src.parkinsons_detection.pipelines.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        # Extract data from the form
        data = CustomData(
            SP_U=float(request.form.get('SP_U')),
            RA_AMP_U=float(request.form.get('RA_AMP_U')),
            LA_AMP_U=float(request.form.get('LA_AMP_U')),
            RA_STD_U=float(request.form.get('RA_STD_U')),
            LA_STD_U=float(request.form.get('LA_STD_U')),
            SYM_U=float(request.form.get('SYM_U')),
            R_JERK_U=float(request.form.get('R_JERK_U')),
            L_JERK_U=float(request.form.get('L_JERK_U')),
            ASA_U=float(request.form.get('ASA_U')),
            ASYM_IND_U=float(request.form.get('ASYM_IND_U')),
            TRA_U=float(request.form.get('TRA_U')),
            T_AMP_U=float(request.form.get('T_AMP_U')),
            CAD_U=float(request.form.get('CAD_U')),
            STR_T_U=float(request.form.get('STR_T_U')),
            STR_CV_U=float(request.form.get('STR_CV_U')),
            STEP_REG_U=float(request.form.get('STEP_REG_U')),
            STEP_SYM_U=float(request.form.get('STEP_SYM_U')),
            JERK_T_U=float(request.form.get('JERK_T_U')),
            SP__DT=float(request.form.get('SP__DT')),
            RA_AMP_DT=float(request.form.get('RA_AMP_DT')),
            LA_AMP_DT=float(request.form.get('LA_AMP_DT')),
            RA_STD_DT=float(request.form.get('RA_STD_DT')),
            LA_STD_DT=float(request.form.get('LA_STD_DT')),
            SYM_DT=float(request.form.get('SYM_DT')),
            R_JERK_DT=float(request.form.get('R_JERK_DT')),
            L_JERK_DT=float(request.form.get('L_JERK_DT')),
            ASA_DT=float(request.form.get('ASA_DT')),
            ASYM_IND_DT=float(request.form.get('ASYM_IND_DT')),
            TRA_DT=float(request.form.get('TRA_DT')),
            T_AMP_DT=float(request.form.get('T_AMP_DT')),
            CAD_DT=float(request.form.get('CAD_DT')),
            STR_T_DT=float(request.form.get('STR_T_DT')),
            STR_CV_DT=float(request.form.get('STR_CV_DT')),
            STEP_REG_DT=float(request.form.get('STEP_REG_DT')),
            STEP_SYM_DT=float(request.form.get('STEP_SYM_DT')),
            JERK_T_DT=float(request.form.get('JERK_T_DT')),
            SW_VEL_OP=float(request.form.get('SW_VEL_OP')),
            SW_PATH_OP=float(request.form.get('SW_PATH_OP')),
            SW_FREQ_OP=float(request.form.get('SW_FREQ_OP')),
            SW_JERK_OP=float(request.form.get('SW_JERK_OP')),
            SW_VEL_CL=float(request.form.get('SW_VEL_CL')),
            SW_PATH_CL=float(request.form.get('SW_PATH_CL')),
            SW_FREQ_CL=float(request.form.get('SW_FREQ_CL')),
            SW_JERK_CL=float(request.form.get('SW_JERK_CL')),
            TUG1_DUR=float(request.form.get('TUG1_DUR')),
            TUG1_STEP_NUM=float(request.form.get('TUG1_STEP_NUM')),
            TUG1_STRAIGHT_DUR=float(request.form.get('TUG1_STRAIGHT_DUR')),
            TUG1_TURNS_DUR=float(request.form.get('TUG1_TURNS_DUR')),
            TUG1_STEP_REG=float(request.form.get('TUG1_STEP_REG')),
            TUG1_STEP_SYM=float(request.form.get('TUG1_STEP_SYM')),
            TUG2_DUR=float(request.form.get('TUG2_DUR')),
            TUG2_STEP_NUM=float(request.form.get('TUG2_STEP_NUM')),
            TUG2_STRAIGHT_DUR=float(request.form.get('TUG2_STRAIGHT_DUR')),
            TUG2_TURNS_DUR=float(request.form.get('TUG2_TURNS_DUR')),
            TUG2_STEP_REG=float(request.form.get('TUG2_STEP_REG')),
            TUG2_STEP_SYM=float(request.form.get('TUG2_STEP_SYM'))
        )
        
        # Convert data to DataFrame
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        # Predict using the pipeline
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print(results)
        if results == 1.0:
            answer = 'Parkinsons found out to be True'
        else:
            answer =   'Parkinsons found out to be False'     
        print("After Prediction")
        
        return render_template('index.html', results=answer)

if __name__ == "__main__":
    app.run(debug=True)
