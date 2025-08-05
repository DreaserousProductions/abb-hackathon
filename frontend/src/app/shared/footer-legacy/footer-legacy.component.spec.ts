import { ComponentFixture, TestBed } from '@angular/core/testing';

import { FooterLegacyComponent } from './footer-legacy.component';

describe('FooterLegacyComponent', () => {
  let component: FooterLegacyComponent;
  let fixture: ComponentFixture<FooterLegacyComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [FooterLegacyComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(FooterLegacyComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
