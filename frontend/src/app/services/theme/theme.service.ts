import { Injectable, Renderer2, RendererFactory2 } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';

export type AppTheme = 'new' | 'legacy';

@Injectable({
  providedIn: 'root'
})
export class ThemeService {
  private renderer: Renderer2;
  private theme$: BehaviorSubject<AppTheme> = new BehaviorSubject<AppTheme>('legacy'); // Changed default to 'legacy'
  private isDark$: BehaviorSubject<boolean> = new BehaviorSubject<boolean>(false);

  constructor(rendererFactory: RendererFactory2) {
    this.renderer = rendererFactory.createRenderer(null, null);
    // Apply the default theme on service initialization
    this.setTheme('legacy');
  }

  /**
   * Toggles between 'new' and 'legacy' themes.
   */
  public toggleTheme(): void {
    const newTheme = this.theme$.value === 'new' ? 'legacy' : 'new';
    this.setTheme(newTheme);
  }

  /**
   * Toggles dark mode (only available in 'new' theme).
   */
  public toggleDarkMode(): void {
    if (this.theme$.value === 'new') {
      const newDarkState = !this.isDark$.value;
      this.isDark$.next(newDarkState);
      this.updateHtmlElementClasses(this.theme$.value);
    }
  }

  /**
   * Returns an Observable for components to subscribe to theme changes.
   */
  public getTheme(): Observable<AppTheme> {
    return this.theme$.asObservable();
  }

  /**
   * Returns an Observable for components to subscribe to dark mode changes.
   */
  public getDarkMode(): Observable<boolean> {
    return this.isDark$.asObservable();
  }

  /**
   * Returns the current theme value.
   */
  public getCurrentTheme(): AppTheme {
    return this.theme$.value;
  }

  /**
   * Returns the current dark mode state.
   */
  public getCurrentDarkMode(): boolean {
    return this.isDark$.value;
  }

  /**
   * Applies the theme changes.
   * @param theme The theme to switch to.
   */
  private setTheme(theme: AppTheme): void {
    // 1. Update the BehaviorSubject to notify subscribers
    this.theme$.next(theme);

    // 2. Reset dark mode when switching to legacy theme
    if (theme === 'legacy') {
      this.isDark$.next(false);
    }

    // 3. Manage the global classes on the <html> tag
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
      // Remove dark class when in legacy theme
      if (htmlElement.classList.contains('dark')) {
        this.renderer.removeClass(htmlElement, 'dark');
      }
    } else { // 'new' theme
      this.renderer.removeClass(htmlElement, 'abb');

      // Handle dark mode for 'new' theme
      if (this.isDark$.value) {
        this.renderer.addClass(htmlElement, 'dark');
      } else {
        this.renderer.removeClass(htmlElement, 'dark');
      }
    }
  }
}
