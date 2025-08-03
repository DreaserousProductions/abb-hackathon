// Models/UserStore.cs
using System.Collections.Generic;
using BCrypt.Net;

namespace ABBHackathon.Models;

public static class UserStore
{
    public static readonly List<User> Users = new()
    {
        new User("admin", BCrypt.Net.BCrypt.HashPassword("admin12."), "Admin"),
        new User("user", BCrypt.Net.BCrypt.HashPassword("user12."), "User")
    };
}
