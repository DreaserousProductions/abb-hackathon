import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { NavigationBarComponent } from "./shared/navigation-bar/navigation-bar.component";
import { FooterComponent } from "./shared/footer/footer.component";
import { NavigationBarLegacyComponent } from "./shared/navigation-bar-legacy/navigation-bar-legacy.component";
import { FooterLegacyComponent } from "./shared/footer-legacy/footer-legacy.component";
import { CommonModule } from '@angular/common';
import { AppTheme, ThemeService } from './services/theme/theme.service';
import { Observable } from 'rxjs';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, NavigationBarComponent, FooterComponent, NavigationBarLegacyComponent, FooterLegacyComponent, CommonModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'frontend';

  public theme$!: Observable<AppTheme>;

  constructor(private themeService: ThemeService) { }

  ngOnInit(): void {
    // Connect the component's observable to the service's observable
    this.theme$ = this.themeService.getTheme();
  }
}
