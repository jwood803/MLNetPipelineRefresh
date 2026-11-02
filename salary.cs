#:package Microsoft.ML@5.0.1
#:property PublishAoT=false

using Microsoft.ML;
using Microsoft.ML.Data;

var context = new MLContext(seed: 42);

// Load data
var data = context.Data.LoadFromTextFile<SalaryData>("./SalaryData.csv", hasHeader: true, separatorChar: ',');

// Split data
var splitData = context.Data.TrainTestSplit(data, testFraction: 0.2);

// Build model
var pipeline = context.Transforms.Concatenate("Features", nameof(SalaryData.YearsExperience))
    .Append(context.Regression.Trainers.Sdca());

var model = pipeline.Fit(splitData.TrainSet);

var testPredictions = model.Transform(splitData.TestSet);

var metrics = context.Regression.Evaluate(testPredictions);

Console.WriteLine($"R^2 - {metrics.RSquared}");
Console.WriteLine($"MAE - {metrics.MeanAbsoluteError}");

context.Model.Save(model, data.Schema, "salary-model.zip");

// Predict
var newData = new SalaryData
{
    YearsExperience = 1.1f
};

var predictionFunc = context.Model.CreatePredictionEngine<SalaryData, SalaryPrediction>(model);

var prediction = predictionFunc.Predict(newData);

Console.WriteLine($"Prediction - {prediction.PredictedSalary}");

public class SalaryData
{
    [LoadColumn(0)]
    public float YearsExperience;

    [LoadColumn(1)]
    [ColumnName("Label")]
    public float Salary;
}

public class SalaryPrediction
{
    [ColumnName("Score")]
    public float PredictedSalary { get; set; }
}
