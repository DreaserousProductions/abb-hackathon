# # Stage 1: build
# FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
# WORKDIR /src
# COPY . .
# RUN dotnet restore
# RUN dotnet publish -c Release -o /app/publish

# # Stage 2: runtime
# FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
# WORKDIR /app
# COPY --from=build /app/publish .
# ENTRYPOINT ["dotnet", "Backend.dll"]

# Stage 1: Build the application
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

# Copy project files and restore dependencies
COPY *.csproj .
RUN dotnet restore

# Copy the rest of the source code and build
COPY . .
RUN dotnet publish -c Release -o /app/publish

# Stage 2: Create the final image
FROM mcr.microsoft.com/dotnet/aspnet:8.0
WORKDIR /app
COPY --from=build /app/publish .

# Expose the port the app runs on
EXPOSE 5000

# Set the entrypoint to run the application
ENTRYPOINT ["dotnet", "backend.dll"]