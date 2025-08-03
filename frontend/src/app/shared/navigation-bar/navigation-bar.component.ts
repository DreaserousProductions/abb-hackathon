import { Component, OnInit } from '@angular/core';
import { RouterLink } from '@angular/router';
import { CommonModule } from '@angular/common';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-navigation-bar',
  standalone: true,
  imports: [RouterLink, CommonModule],
  templateUrl: './navigation-bar.component.html',
  styleUrl: './navigation-bar.component.css'
})
export class NavigationBarComponent implements OnInit {
  focusLink: string = 'Login';

  ngOnInit(): void {
    const hamburg = document.querySelector(".hamburger-btn") as HTMLDivElement;
    hamburg.addEventListener("click", this.toggleMobileMenu);
  }

  toggleMobileMenu(): void {
    const mobMenu = document.querySelector(".navigation-bar") as HTMLDivElement;
    mobMenu.classList.toggle("active");
  }
}