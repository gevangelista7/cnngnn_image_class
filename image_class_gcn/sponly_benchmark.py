import joblib

files = ['TESTED_LinearModel_2023-07-25T20_12_42.765462',
         'TESTED_LinearModel_2023-07-25T20_23_36.373658',
         'TESTED_LinearModel_2023-07-25T20_35_12.517717',
         'TESTED_LinearModel_2023-07-25T20_47_10.827562',
         'TESTED_LinearModel_2023-07-25T21_20_05.083234',
         'TESTED_LinearModel_2023-07-25T22_05_38.131221',
         'TESTED_LinearModel_2023-07-25T22_17_34.902672',
         'TESTED_LinearModel_2023-07-25T22_34_53.346905',
         'TESTED_LinearModel_2023-07-25T22_47_40.512325',
         'TESTED_LinearModel_2023-07-25T23_00_02.815218',
         'TESTED_LinearModel_2023-07-25T23_13_28.203635',
         'TESTED_LinearModel_2023-07-25T23_31_03.870744',
         'TESTED_LinearModel_2023-07-25T23_43_43.061518',
         'TESTED_LinearModel_2023-07-25T23_56_07.978623',
         'TESTED_LinearModel_2023-07-26T00_09_35.123438',
         'TESTED_LinearModel_2023-07-26T00_27_25.079463',
         'TESTED_LinearModel_2023-07-26T00_40_03.939536',
         'TESTED_LinearModel_2023-07-26T00_52_50.781153',
         'TESTED_LinearModel_2023-07-26T01_06_18.471088',
         'TESTED_LinearModel_2023-07-26T06_44_39.989265',
         'TESTED_LinearModel_2023-07-26T06_52_39.164627',
         'TESTED_LinearModel_2023-07-26T07_03_13.560598',
         'TESTED_LinearModel_2023-07-26T07_14_53.734600',
         'TESTED_LinearModel_2023-07-26T07_30_48.404929',
         'TESTED_LinearModel_2023-07-26T07_42_18.072769',
         'TESTED_LinearModel_2023-07-26T07_53_37.387424',
         'TESTED_LinearModel_2023-07-26T08_06_11.585403']

if __name__ == '__main__':

    for file in files:
        result = joblib.load('./results/'+file)
        print(result.attrs['preprocess_file'])
        print(result.attrs['test_acc'])
