import {
  Component,
  AfterViewInit,
  ElementRef,
  Renderer2
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { AuthService } from '../../services/auth/auth.service';
import { HttpClientModule } from '@angular/common/http';

interface LoginData {
  username: string;
  password: string;
}

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [FormsModule, HttpClientModule],
  templateUrl: './login.component.html',
  styleUrl: './login.component.css'
})
export class LoginComponent implements AfterViewInit {
  loginData: LoginData = {
    username: '',
    password: ''
  };

  isLoading: boolean = false;

  constructor(
    private router: Router,
    private elementRef: ElementRef,
    private renderer: Renderer2,
    private authService: AuthService
  ) { }

  ngAfterViewInit(): void {
    this.initializeFocusEffects();
  }

  onSubmit(): void {
    if (this.isValidForm()) {
      this.isLoading = true;

      // Simulate login process
      setTimeout(() => {
        this.performLogin();
      }, 1500);
    }
  }

  private isValidForm(): boolean {
    return this.loginData.username.trim() !== '' &&
      this.loginData.password.trim() !== '';
  }

  private performLogin(): void {
    this.authService.login(this.loginData.username, this.loginData.password)
      .subscribe({
        next: (response) => {
          console.log('Login successful:', response);

          // Store token if needed
          localStorage.setItem('jwt', response.token);

          // Navigate to dashboard
          this.router.navigate(['/dashboard']);
          this.isLoading = false;
        },
        error: (err) => {
          console.error('Login failed:', err);
          this.handleLoginError();
          this.isLoading = false;
        }
      });
  }

  private handleLoginError(): void {
    // Add error handling logic here
    console.error('Login failed');
    // You could show an error message or toast
  }

  private initializeFocusEffects(): void {
    const inputs = this.elementRef.nativeElement.querySelectorAll('input');

    inputs.forEach((input: HTMLInputElement) => {
      this.renderer.listen(input, 'focus', () => {
        this.renderer.setStyle(input.parentElement, 'transform', 'scale(1.01)');
      });

      this.renderer.listen(input, 'blur', () => {
        this.renderer.setStyle(input.parentElement, 'transform', 'scale(1)');
      });
    });
  }
}