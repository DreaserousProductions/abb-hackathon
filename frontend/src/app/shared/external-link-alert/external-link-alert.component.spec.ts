import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ExternalLinkAlertComponent } from './external-link-alert.component';

describe('ExternalLinkAlertComponent', () => {
  let component: ExternalLinkAlertComponent;
  let fixture: ComponentFixture<ExternalLinkAlertComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ExternalLinkAlertComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ExternalLinkAlertComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
