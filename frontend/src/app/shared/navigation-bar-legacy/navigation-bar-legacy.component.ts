import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';

@Component({
  selector: 'app-navigation-bar-legacy',
  standalone: true,
  imports: [RouterLink],
  templateUrl: './navigation-bar-legacy.component.html',
  styleUrl: './navigation-bar-legacy.component.css'
})
export class NavigationBarLegacyComponent {

  ngOnInit(): void {
    const hamburg = document.querySelector(".hamburger-btn") as HTMLDivElement;
    if (hamburg) {
      hamburg.addEventListener("click", this.toggleMobileMenu);
    }
  }

  toggleMobileMenu(): void {
    const mobMenu = document.querySelector(".navigation-bar") as HTMLDivElement;
    if (mobMenu) {
      mobMenu.classList.toggle("active");
    }
  }
}
