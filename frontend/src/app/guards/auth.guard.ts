// src/app/auth.guard.ts
import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { of } from 'rxjs';
import { catchError, map } from 'rxjs/operators';

function isTokenExpired(token: string): boolean {
  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    const now = Math.floor(Date.now() / 1000);
    return payload.exp < now;
  } catch {
    return true;
  }
}

let redirecting = false;
function safeRedirect(router: Router, path: string) {
  if (!redirecting) {
    redirecting = true;
    router.navigate([path]).then(() => redirecting = false);
  }
}

export const authGuard: CanActivateFn = (route, state) => {
  const http = inject(HttpClient);
  const router = inject(Router);

  const token = localStorage.getItem('authToken');

  if (!token || isTokenExpired(token)) {
    localStorage.removeItem('authToken');
    safeRedirect(router, '/login');
    return false;
  }

  const headers = new HttpHeaders({
    'Authorization': `Bearer ${token}`
  });

  const verifyUrl = 'http://localhost:5000/api/Auth/verify';

  return http.get<any>(verifyUrl, { headers }).pipe(
    map(() => true),
    catchError(() => {
      localStorage.removeItem('authToken');
      safeRedirect(router, '/login');
      return of(false);
    })
  );
};
