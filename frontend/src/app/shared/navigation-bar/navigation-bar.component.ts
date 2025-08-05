import { Component, HostBinding, OnDestroy, OnInit } from '@angular/core';
import { RouterLink } from '@angular/router';
import { CommonModule } from '@angular/common';
import { Subscription } from 'rxjs';
import { AppTheme, ThemeService } from '../../services/theme/theme.service';

@Component({
  selector: 'app-navigation-bar',
  standalone: true,
  imports: [RouterLink, CommonModule],
  templateUrl: './navigation-bar.component.html',
  styleUrl: './navigation-bar.component.css'
})
export class NavigationBarComponent implements OnInit, OnDestroy {
  focusLink: string = 'Login';

  // NEW: Add a property to track if the menu is open or closed.
  isMenuOpen = false;

  constructor(private themeService: ThemeService) { }

  @HostBinding('class') themeClass: string = 'theme-new';
  private themeSubscription!: Subscription;

  ngOnInit(): void {
    // Subscribe to theme changes
    this.themeSubscription = this.themeService.getTheme().subscribe((theme: AppTheme) => {
      // Update the host class based on the current theme
      this.themeClass = theme === 'legacy' ? 'theme-legacy' : 'theme-new';
    });

    // REMOVED: All document.querySelector and addEventListener calls are gone.
  }

  ngOnDestroy(): void {
    if (this.themeSubscription) {
      this.themeSubscription.unsubscribe();
    }
  }

  // UPDATED: This method now simply toggles the boolean property.
  toggleMobileMenu(): void {
    this.isMenuOpen = !this.isMenuOpen;
  }

  onLegacyButtonClick(): void {
    this.themeService.toggleTheme();
  }
}