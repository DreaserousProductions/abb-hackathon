import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';
import { ThemeService } from '../../services/theme/theme.service';

@Component({
  selector: 'app-navigation-bar-legacy',
  standalone: true,
  imports: [RouterLink],
  templateUrl: './navigation-bar-legacy.component.html',
  styleUrl: './navigation-bar-legacy.component.css'
})
export class NavigationBarLegacyComponent {

  constructor(private themeService: ThemeService) { }

  ngOnInit(): void {
    const hamburg = document.querySelector(".hamburger-btn") as HTMLDivElement;
    if (hamburg) {
      hamburg.addEventListener("click", this.toggleMobileMenu);
    }
  }

  onToggleTheme() {
    this.themeService.toggleTheme();
  }

  toggleMobileMenu(): void {
    const mobMenu = document.querySelector(".navigation-bar") as HTMLDivElement;
    if (mobMenu) {
      mobMenu.classList.toggle("active");
    }
  }
}
