using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System.Text;
using System.Text.Json;

namespace ABBHackathon.Controllers;

[ApiController]
[Route("api/[controller]")]
public class ModelController : ControllerBase
{
    private readonly IHttpClientFactory _httpClientFactory;
    // In a real app, this would come from IConfiguration
    private const string FastAPI_BaseUrl = "http://localhost:8000"; 

    public ModelController(IHttpClientFactory httpClientFactory)
    {
        _httpClientFactory = httpClientFactory;
    }

    [Authorize]
    [HttpPost("train")]
    public async Task<IActionResult> Train([FromBody] TrainModelRequest request)
    {
        if (request == null || string.IsNullOrEmpty(request.UserId) || string.IsNullOrEmpty(request.DatasetId))
        {
            return BadRequest("UserId and DatasetId are required.");
        }

        try
        {
            var client = _httpClientFactory.CreateClient();

            // Serialize the request with camelCase naming to match FastAPI's Pydantic models
            var jsonContent = JsonSerializer.Serialize(request, new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });
            
            var httpContent = new StringContent(jsonContent, Encoding.UTF8, "application/json");

            var response = await client.PostAsync($"{FastAPI_BaseUrl}/csv/train", httpContent);

            var responseContent = await response.Content.ReadAsStringAsync();

            if (!response.IsSuccessStatusCode)
            {
                // Forward the detailed error from the FastAPI backend
                return StatusCode((int)response.StatusCode, responseContent);
            }

            // Deserialize the successful response from FastAPI
            var result = JsonSerializer.Deserialize<TrainModelResponse>(responseContent, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });

            return Ok(result);
        }
        catch (Exception ex)
        {
            // Log the exception ex
            return StatusCode(500, $"Internal server error while communicating with the training service: {ex.Message}");
        }
    }
}

// --- MODELS FOR THIS CONTROLLER ---
// You can place these in the same file or a separate Models folder

public class TrainModelRequest
{
    public string UserId { get; set; } = null!;
    public string DatasetId { get; set; } = null!;
    public Dictionary<string, DateRangeModel> DateRanges { get; set; } = new();
}

public class TrainModelResponse
{
    public TrainingMetrics? Metrics { get; set; }
    public TrainingPlots? Plots { get; set; }
}

public class TrainingMetrics
{
    public float Accuracy { get; set; }
    public float Precision { get; set; }
    public float Recall { get; set; }
    public float F1Score { get; set; }

    public int TruePositive { get; set; }
    public int FalsePositive { get; set; }
    public int TrueNegative { get; set; }
    public int FalseNegative { get; set; }
}

public class TrainingPlots
{
    public string? FeatureImportance { get; set; }
}