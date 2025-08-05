import { Injectable, Renderer2, RendererFactory2 } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';

export type AppTheme = 'new' | 'legacy';

@Injectable({
  providedIn: 'root'
})
export class ThemeService {
  private renderer: Renderer2;
  private theme$: BehaviorSubject<AppTheme> = new BehaviorSubject<AppTheme>('new');

  constructor(rendererFactory: RendererFactory2) {
    this.renderer = rendererFactory.createRenderer(null, null);
  }

  /**
   * Toggles the theme and applies all necessary changes.
   */
  public toggleTheme(): void {
    const newTheme = this.theme$.value === 'new' ? 'legacy' : 'new';
    this.setTheme(newTheme);
  }

  /**
   * Returns an Observable for components to subscribe to theme changes.
   */
  public getTheme(): Observable<AppTheme> {
    return this.theme$.asObservable();
  }

  /**
   * Applies the theme changes.
   * @param theme The theme to switch to.
   */
  private setTheme(theme: AppTheme): void {
    // 1. Update the BehaviorSubject to notify subscribers (like AppComponent and UploadComponent)
    this.theme$.next(theme);

    // 2. Manage the global classes on the <html> tag
    this.updateHtmlElementClasses(theme);
  }

  /**
   * Manages classes on the root <html> element.
   * @param theme The active theme.
   */
  private updateHtmlElementClasses(theme: AppTheme): void {
    const htmlElement = document.documentElement; // Get the <html> element

    if (theme === 'legacy') {
      this.renderer.addClass(htmlElement, 'abb');
      // Ensure .dark is removed if it exists
      if (htmlElement.classList.contains('dark')) {
        this.renderer.removeClass(htmlElement, 'dark');
      }
    } else { // 'new' theme
      this.renderer.removeClass(htmlElement, 'abb');
      // Here you can decide if the 'new' theme should have the .dark class by default
    }
  }
}