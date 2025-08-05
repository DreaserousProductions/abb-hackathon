using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;

namespace ABBHackathon.Controllers;

[ApiController]
[Route("api/[controller]")]
public class ModelController : ControllerBase
{
    private readonly IHttpClientFactory _httpClientFactory;
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

    // ===================================================================
    // NEW SIMULATION WEBSOCKET ENDPOINT
    // ===================================================================
    [HttpGet("simulation-ws")]
    public async Task GetSimulationStream()
    {
        // This endpoint must be accessed via a WebSocket request (ws:// or wss://)
        if (HttpContext.WebSockets.IsWebSocketRequest)
        {
            // Accept the WebSocket connection from the Angular frontend
            using var angularSocket = await HttpContext.WebSockets.AcceptWebSocketAsync();
            
            // Establish a new WebSocket connection to the FastAPI backend
            using var fastApiSocket = new ClientWebSocket();
            var fastApiUrl = new Uri(FastAPI_BaseUrl.Replace("http", "ws") + "/csv/simulation-ws");
            
            await fastApiSocket.ConnectAsync(fastApiUrl, CancellationToken.None);

            // Create two tasks to proxy messages in both directions
            var angularToFastApi = Proxy(angularSocket, fastApiSocket);
            var fastApiToAngular = Proxy(fastApiSocket, angularSocket);

            // Run both proxy tasks concurrently until one of them completes (e.g., a socket closes)
            await Task.WhenAny(angularToFastApi, fastApiToAngular);

            // Ensure both sockets are closed gracefully
            if (angularSocket.State == WebSocketState.Open)
                await angularSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Closing", CancellationToken.None);
            if (fastApiSocket.State == WebSocketState.Open)
                await fastApiSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Closing", CancellationToken.None);
        }
        else
        {
            // If accessed via a normal HTTP GET, return a Bad Request
            HttpContext.Response.StatusCode = StatusCodes.Status400BadRequest;
        }
    }

    /// <summary>
    /// A helper method to continuously proxy messages from a source socket to a destination socket.
    /// </summary>
    private static async Task Proxy(WebSocket source, WebSocket destination)
    {
        var buffer = new byte[1024 * 4];
        try
        {
            while (source.State == WebSocketState.Open && destination.State == WebSocketState.Open)
            {
                var result = await source.ReceiveAsync(new ArraySegment<byte>(buffer), CancellationToken.None);
                
                if (result.MessageType == WebSocketMessageType.Close)
                {
                    // If one socket sends a close message, forward it to the other.
                    await destination.CloseAsync(result.CloseStatus ?? WebSocketCloseStatus.NormalClosure, result.CloseStatusDescription, CancellationToken.None);
                    break;
                }

                // Forward the received message to the destination socket.
                await destination.SendAsync(new ArraySegment<byte>(buffer, 0, result.Count), result.MessageType, result.EndOfMessage, CancellationToken.None);
            }
        }
        catch (WebSocketException ex)
        {
            // Handle cases where a socket is closed abruptly.
            Console.WriteLine($"WebSocket proxy error: {ex.Message}");
        }
    }
}

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

// public class DateRangeModel { public string? Start { get; set; } public string? End { get; set; } } // Questionable