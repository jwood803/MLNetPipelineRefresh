#:include SalaryData.cs
#:include SalaryPrediction.cs
#:package Microsoft.ML@5.0.1
#:property PublishAoT=false

using Microsoft.ML;

var context = new MLContext(seed: 42);

// Load data
var data = context.Data.LoadFromTextFile<SalaryData>("./SalaryData.csv", hasHeader: true, separatorChar: ',');

// Split data
var splitData = context.Data.TrainTestSplit(data, testFraction: 0.2);

// Build model
var pipeline = context.Transforms.Concatenate("Features", nameof(SalaryData.YearsExperience))
    .Append(context.Regression.Trainers.Sdca(labelColumnName: nameof(SalaryData.Salary)));

var model = pipeline.Fit(splitData.TrainSet);

var testPredictions = model.Transform(splitData.TestSet);

// Run this with test set to show that the data needs to be transformed first and labelcolumnname is needed as it defaults to "Label"
var metrics = context.Regression.Evaluate(testPredictions, labelColumnName: nameof(SalaryData.Salary));

Console.WriteLine($"R^2 - {metrics.RSquared}");
Console.WriteLine($"MAE - {metrics.MeanAbsoluteError}");

context.Model.Save(model, data.Schema, "salary-model.zip");

// Predict
var newData = new SalaryData
{
    YearsExperience = 18f
};

var predictionFunc = context.Model.CreatePredictionEngine<SalaryData, SalaryPrediction>(model);

var prediction = predictionFunc.Predict(newData);

Console.WriteLine($"Prediction - {prediction.PredictedSalary}");
