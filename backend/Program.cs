// using Microsoft.AspNetCore.Authentication.JwtBearer;
// using Microsoft.IdentityModel.Tokens;
// using Microsoft.AspNetCore.Http.Features;
// using System.Text;

// var builder = WebApplication.CreateBuilder(args);

// builder.Services.AddControllers();

// // Add services to the container.
// // Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
// builder.Services.AddEndpointsApiExplorer();
// builder.Services.AddSwaggerGen();

// var allowedOrigins = new List<string> { "http://localhost:4200", "http://localhost" };

// builder.Services.AddCors(options =>
// {
//     options.AddPolicy("AllowAngularApp", policy =>
//     {
//         policy.SetIsOriginAllowed(origin => 
//               {
//                   return allowedOrigins.Contains(origin);
//               })
//               .AllowAnyHeader()
//               .AllowAnyMethod()
//               .AllowCredentials(); 
//     });
// });

// builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
//     .AddJwtBearer(options =>
//     {
//         options.TokenValidationParameters = new TokenValidationParameters
//         {
//             ValidateIssuer = true,
//             ValidateAudience = true,
//             ValidateLifetime = true,
//             ValidateIssuerSigningKey = true,
//             ValidIssuer = builder.Configuration["Jwt:Issuer"],
//             ValidAudience = builder.Configuration["Jwt:Audience"],
//             IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(builder.Configuration["Jwt:Key"]!))
//         };
//     });

// builder.Services.AddAuthorization();

// // builder.Services.Configure<FormOptions>(options =>
// // {
// //     options.MultipartBodyLengthLimit = long.MaxValue; // or specify a size like 2L * 1024 * 1024 * 1024 (2GB)
// // });

// builder.WebHost.UseUrls("http://0.0.0.0:5000");
// builder.Services.AddHttpClient();
// var app = builder.Build();

// // Configure the HTTP request pipeline.
// if (app.Environment.IsDevelopment())
// {
//     app.UseSwagger();
//     app.UseSwaggerUI();
// }

// // app.UseHttpsRedirection();

// app.UseWebSockets();

// app.UseCors("AllowAngularApp"); // Apply CORS policy
// app.UseAuthentication();        // Enable authentication
// app.UseAuthorization();         // Enable authorization

// app.MapControllers();

// app.Run();
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.IdentityModel.Tokens;
using Microsoft.AspNetCore.Http.Features;
using System.Text;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();

// Add services to the container.
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var allowedOrigins = new List<string> { "http://localhost:4200", "http://localhost" };

builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowAngularApp", policy =>
    {
        policy.SetIsOriginAllowed(origin =>
        {
            return allowedOrigins.Contains(origin);
        })
        .AllowAnyHeader()
        .AllowAnyMethod()
        .AllowCredentials();
    });
});

builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
    .AddJwtBearer(options =>
    {
        options.TokenValidationParameters = new TokenValidationParameters
        {
            ValidateIssuer = true,
            ValidateAudience = true,
            ValidateLifetime = true,
            ValidateIssuerSigningKey = true,
            ValidIssuer = builder.Configuration["Jwt:Issuer"],
            ValidAudience = builder.Configuration["Jwt:Audience"],
            IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(builder.Configuration["Jwt:Key"]!))
        };
    });

builder.Services.AddAuthorization();

builder.WebHost.UseUrls("http://0.0.0.0:5000");

// Keep existing default HttpClient (preserves existing functionality)
builder.Services.AddHttpClient();

// Environment-based ML Service configuration
var mlServiceBaseUrl = builder.Environment.IsDevelopment() 
    ? "http://localhost:8000"  // Development
    : "http://ml_service:8000"; // Docker/Production

// Add new named HttpClient with 3-minute timeout for ML service calls
builder.Services.AddHttpClient("MLService", httpClient =>
{
    httpClient.BaseAddress = new Uri(mlServiceBaseUrl);
    httpClient.Timeout = TimeSpan.FromMinutes(3); // 3-minute timeout
});

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseWebSockets();
app.UseCors("AllowAngularApp"); // Apply CORS policy
app.UseAuthentication(); // Enable authentication
app.UseAuthorization(); // Enable authorization
app.MapControllers();

app.Run();
