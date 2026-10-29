using Microsoft.ML.Data;

public class SalaryData
{
    [LoadColumn(0)]
    public float YearsExperience;

    [LoadColumn(1)]
    public float Salary;
}
