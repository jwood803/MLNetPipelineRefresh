using Microsoft.ML.Data;

public class SalaryPrediction
{
    [ColumnName("Score")]
    public float PredictedSalary { get; set; }
}
