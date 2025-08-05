// ABBHackathon/Controllers/DataController.cs

using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace ABBHackathon.Controllers;

[ApiController]
[Route("api/[controller]")]
public class DataController : ControllerBase
{
    private readonly IHttpClientFactory _httpClientFactory;
    // In a real app, this would come from configuration
    private const string FastAPI_BaseUrl = "http://localhost:8000";

    public DataController(IHttpClientFactory httpClientFactory)
    {
        _httpClientFactory = httpClientFactory;
    }

    [Authorize]
    [HttpPost("upload")]
    public async Task<IActionResult> UploadFile([FromForm] FileUploadModel model)
    {
        var file = model.File;
        if (file == null || file.Length == 0)
            return BadRequest("No file uploaded.");
        
        if (string.IsNullOrEmpty(model.UserId))
            return BadRequest("User ID is required.");

        try
        {
            using var content = new MultipartFormDataContent();
            var streamContent = new StreamContent(file.OpenReadStream());
            streamContent.Headers.ContentType = new MediaTypeHeaderValue(file.ContentType);
            content.Add(streamContent, "file", file.FileName);
            content.Add(new StringContent(model.UserId), "user_id");

            var client = _httpClientFactory.CreateClient();
            var response = await client.PostAsync($"{FastAPI_BaseUrl}/csv/process", content);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                return StatusCode((int)response.StatusCode, $"FastAPI processing failed: {errorContent}");
            }

            var responseContent = await response.Content.ReadAsStringAsync();
            var result = JsonSerializer.Deserialize<UploadResult>(responseContent, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
            return Ok(result);
        }
        catch (Exception ex)
        {
            return StatusCode(500, $"Internal server error: {ex.Message}");
        }
    }

    [Authorize]
    [HttpPost("validate-ranges")]
    public async Task<IActionResult> ValidateRanges([FromBody] ValidateRangesRequest request)
    {
        try
        {
            var client = _httpClientFactory.CreateClient();
            
            // --- FIX IS HERE: Use CamelCase for serialization to match FastAPI's model ---
            var jsonContent = JsonSerializer.Serialize(request, new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });
            // --- END OF FIX ---

            var httpContent = new StringContent(jsonContent, Encoding.UTF8, "application/json");

            var response = await client.PostAsync($"{FastAPI_BaseUrl}/csv/validate-ranges", httpContent);

            var responseContent = await response.Content.ReadAsStringAsync();

            if (!response.IsSuccessStatusCode)
            {
                // Forward the error from FastAPI
                return StatusCode((int)response.StatusCode, responseContent);
            }

            var result = JsonSerializer.Deserialize<ValidateRangesResponse>(responseContent, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });

            return Ok(result);
        }
        catch (Exception ex)
        {
            return StatusCode(500, $"Internal server error: {ex.Message}");
        }
    }
}

// --- MODELS FOR REQUESTS AND RESPONSES ---

public class FileUploadModel
{
    public IFormFile File { get; set; } = null!;
    public string UserId { get; set; } = null!;
}

public class UploadResult
{
    public string? DatasetId { get; set; }
    public string? UserId { get; set; }
    public int TotalRecords { get; set; }
    public int NumColumns { get; set; }
    public float PassRate { get; set; }
    public DateRangeModel? DateRange { get; set; }
}

public class DateRangeModel
{
    public string? Start { get; set; }
    public string? End { get; set; }
}

public class ValidateRangesRequest
{
    public string UserId { get; set; } = null!;
    public string DatasetId { get; set; } = null!;
    public Dictionary<string, DateRangeModel> DateRanges { get; set; } = new();
}

public class ValidateRangesResponse
{
    public string Status { get; set; } = null!;
    public RangeCount? Training { get; set; }
    public RangeCount? Testing { get; set; }
    public RangeCount? Simulation { get; set; }
    public Dictionary<string, int>? MonthlyCounts { get; set; }
}

public class RangeCount
{
    public int Count { get; set; }
}
