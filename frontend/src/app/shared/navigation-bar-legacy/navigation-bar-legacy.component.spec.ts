import { ComponentFixture, TestBed } from '@angular/core/testing';

import { NavigationBarLegacyComponent } from './navigation-bar-legacy.component';

describe('NavigationBarLegacyComponent', () => {
  let component: NavigationBarLegacyComponent;
  let fixture: ComponentFixture<NavigationBarLegacyComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [NavigationBarLegacyComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(NavigationBarLegacyComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
