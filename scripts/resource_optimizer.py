"""
Hospital Resource Optimizer
Optimizes staff allocation, bed management, and emergency preparedness
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import config

class HospitalResourceOptimizer:
    """Optimize hospital resources based on predictions"""
    
    def __init__(self):
        self.icu_capacity = config.ICU_CAPACITY
        self.critical_threshold = config.CRITICAL_THRESHOLD
        self.staff_ratio = config.STAFF_PER_10_PATIENTS
        
    def calculate_staff_requirements(self, predicted_admissions, predicted_icu, predicted_workload):
        """
        Calculate optimal staff allocation
        
        Args:
            predicted_admissions: Array of predicted emergency admissions
            predicted_icu: Array of predicted ICU demand
            predicted_workload: Array of predicted staff workload
            
        Returns:
            Dictionary with staff recommendations
        """
        # Base staff from emergency admissions
        base_staff = np.ceil(predicted_admissions * self.staff_ratio / 10)
        
        # Additional staff for ICU (ICU needs 2x staff ratio)
        icu_staff = np.ceil(predicted_icu * self.staff_ratio / 5)
        
        # Workload-based adjustment (already calculated by predictor)
        workload_staff = predicted_workload
        
        # Take maximum to be safe
        total_staff = np.maximum(base_staff, workload_staff)
        total_staff += icu_staff * 0.5  # ICU staff partially overlaps
        
        # Round up and ensure minimum
        total_staff = np.ceil(total_staff).astype(int)
        total_staff = np.maximum(total_staff, 5)  # Minimum 5 staff always
        
        # Categorize by shift (assuming 8-hour shifts)
        shifts = []
        for i in range(0, len(total_staff), 8):
            shift_staff = total_staff[i:i+8].mean()
            shifts.append({
                'shift_start_hour': i,
                'required_staff': int(np.ceil(shift_staff)),
                'peak_hour_staff': int(total_staff[i:i+8].max()),
                'min_hour_staff': int(total_staff[i:i+8].min())
            })
        
        return {
            'hourly_staff': total_staff,
            'shifts': shifts,
            'total_staff_24h': int(total_staff[:24].sum()),
            'peak_staff': int(total_staff.max()),
            'avg_staff': float(total_staff.mean())
        }
    
    def assess_bed_capacity(self, predicted_admissions, predicted_icu, current_occupancy=50):
        """
        Assess bed capacity and recommend actions
        
        Args:
            predicted_admissions: Array of predicted admissions
            predicted_icu: Array of predicted ICU demand
            current_occupancy: Current bed occupancy
            
        Returns:
            Dictionary with bed management recommendations
        """
        # Estimate bed turnover (assume 24-hour average stay)
        hourly_turnover = current_occupancy / 24
        
        # Projected occupancy
        net_admissions = predicted_admissions - hourly_turnover
        projected_occupancy = current_occupancy + np.cumsum(net_admissions)
        
        # ICU capacity check
        icu_utilization = predicted_icu / self.icu_capacity
        
        # Identify critical periods
        critical_hours = np.where(icu_utilization > self.critical_threshold)[0]
        
        recommendations = {
            'projected_occupancy': projected_occupancy,
            'icu_utilization': icu_utilization,
            'critical_hours': critical_hours.tolist(),
            'max_icu_utilization': float(icu_utilization.max()),
            'avg_icu_utilization': float(icu_utilization.mean())
        }
        
        # Generate alerts
        alerts = []
        if len(critical_hours) > 0:
            alerts.append({
                'severity': 'HIGH',
                'message': f'ICU capacity will exceed {self.critical_threshold*100:.0f}% in {len(critical_hours)} hours',
                'action': 'Prepare overflow ICU beds, contact additional ICU staff'
            })
        
        if icu_utilization.max() > 0.95:
            alerts.append({
                'severity': 'CRITICAL',
                'message': 'ICU capacity near maximum',
                'action': 'Activate emergency protocols, consider patient transfers'
            })
        
        if predicted_admissions.sum() > predicted_admissions.mean() * 48 * 1.5:
            alerts.append({
                'severity': 'MEDIUM',
                'message': 'Unusually high admission volume predicted',
                'action': 'Increase emergency department staff, prepare additional beds'
            })
        
        recommendations['alerts'] = alerts
        return recommendations
    
    def create_emergency_preparedness_plan(self, staff_req, bed_assess):
        """
        Create comprehensive emergency preparedness plan
        
        Args:
            staff_req: Staff requirements dictionary
            bed_assess: Bed assessment dictionary
            
        Returns:
            Formatted preparedness plan
        """
        plan = {
            'timestamp': datetime.now().isoformat(),
            'planning_horizon': '48 hours',
            'status': 'NORMAL',
            'recommendations': []
        }
        
        # Determine status
        if bed_assess['max_icu_utilization'] > 0.95:
            plan['status'] = 'CRITICAL'
        elif bed_assess['max_icu_utilization'] > self.critical_threshold or len(bed_assess['alerts']) > 0:
            plan['status'] = 'ELEVATED'
        
        # Staff recommendations
        plan['recommendations'].append({
            'category': 'Staffing',
            'priority': 'HIGH',
            'details': [
                f"Peak staff requirement: {staff_req['peak_staff']} personnel",
                f"Average staff needed: {staff_req['avg_staff']:.1f} personnel per hour",
                f"Total staff-hours (24h): {staff_req['total_staff_24h']} hours",
                f"Recommended shift configuration: {len(staff_req['shifts'])} shifts of 8 hours"
            ]
        })
        
        # Bed management
        plan['recommendations'].append({
            'category': 'Bed Management',
            'priority': 'HIGH' if bed_assess['max_icu_utilization'] > self.critical_threshold else 'MEDIUM',
            'details': [
                f"Expected peak ICU utilization: {bed_assess['max_icu_utilization']*100:.1f}%",
                f"Average ICU utilization: {bed_assess['avg_icu_utilization']*100:.1f}%",
                f"Critical periods: {len(bed_assess['critical_hours'])} hours above threshold"
            ]
        })
        
        # Add alerts
        if bed_assess['alerts']:
            plan['recommendations'].append({
                'category': 'Alerts',
                'priority': 'URGENT',
                'details': [alert['message'] + ' → ' + alert['action'] for alert in bed_assess['alerts']]
            })
        
        # Resource mobilization
        if plan['status'] != 'NORMAL':
            plan['recommendations'].append({
                'category': 'Resource Mobilization',
                'priority': 'HIGH',
                'details': [
                    'Contact on-call staff for potential overtime',
                    'Review supply inventory (PPE, medications, equipment)',
                    'Coordinate with neighboring hospitals for transfer capacity',
                    'Activate incident command if status escalates to CRITICAL'
                ]
            })
        
        return plan
    
    def optimize(self, predicted_admissions, predicted_icu, predicted_workload, current_occupancy=50):
        """
        Main optimization function - coordinates all optimization tasks
        
        Returns:
            Complete optimization results and recommendations
        """
        # Calculate staff requirements
        staff_req = self.calculate_staff_requirements(
            predicted_admissions, 
            predicted_icu, 
            predicted_workload
        )
        
        # Assess bed capacity
        bed_assess = self.assess_bed_capacity(
            predicted_admissions,
            predicted_icu,
            current_occupancy
        )
        
        # Create preparedness plan
        preparedness_plan = self.create_emergency_preparedness_plan(
            staff_req,
            bed_assess
        )
        
        return {
            'staff_requirements': staff_req,
            'bed_assessment': bed_assess,
            'preparedness_plan': preparedness_plan
        }

def demo():
    """Demo the optimizer with sample predictions"""
    print("=== Hospital Resource Optimizer Demo ===\n")
    
    # Load sample predictions (or create synthetic ones)
    # Simulate 48 hours of predictions
    np.random.seed(42)
    
    hours = 48
    predicted_admissions = np.random.poisson(2.5, hours)  # ~2-3 per hour
    predicted_icu = np.random.poisson(0.4, hours)  # ~15% need ICU
    predicted_workload = predicted_admissions * 1.2 + np.random.randint(0, 3, hours)
    
    # Add a surge event at hour 30-36
    predicted_admissions[30:36] *= 2
    predicted_icu[30:36] = np.clip(predicted_icu[30:36] * 2, 0, config.ICU_CAPACITY)
    
    print(f"Simulating predictions for next {hours} hours")
    print(f"Expected admissions: {predicted_admissions.sum()}")
    print(f"Expected ICU demand: {predicted_icu.sum()}")
    print(f"Peak hour admissions: {predicted_admissions.max()}\n")
    
    # Optimize
    optimizer = HospitalResourceOptimizer()
    results = optimizer.optimize(
        predicted_admissions,
        predicted_icu,
        predicted_workload,
        current_occupancy=60
    )
    
    # Display results
    print("=== OPTIMIZATION RESULTS ===\n")
    
    print(f"Status: {results['preparedness_plan']['status']}")
    print(f"\n--- Staff Requirements ---")
    print(f"Peak staff needed: {results['staff_requirements']['peak_staff']} personnel")
    print(f"Average staff per hour: {results['staff_requirements']['avg_staff']:.1f}")
    print(f"Total staff-hours (24h): {results['staff_requirements']['total_staff_24h']}")
    
    print(f"\n--- Shift Recommendations ---")
    for shift in results['staff_requirements']['shifts'][:3]:  # First 3 shifts
        print(f"  Hour {shift['shift_start_hour']:02d}-{shift['shift_start_hour']+7:02d}: "
              f"{shift['required_staff']} staff (peak: {shift['peak_hour_staff']})")
    
    print(f"\n--- Bed Management ---")
    print(f"Max ICU utilization: {results['bed_assessment']['max_icu_utilization']*100:.1f}%")
    print(f"Avg ICU utilization: {results['bed_assessment']['avg_icu_utilization']*100:.1f}%")
    print(f"Critical hours: {len(results['bed_assessment']['critical_hours'])}")
    
    if results['bed_assessment']['alerts']:
        print(f"\n--- ALERTS ---")
        for alert in results['bed_assessment']['alerts']:
            print(f"  [{alert['severity']}] {alert['message']}")
            print(f"    → {alert['action']}")
    
    print(f"\n--- Recommendations ---")
    for rec in results['preparedness_plan']['recommendations']:
        print(f"\n{rec['category']} (Priority: {rec['priority']})")
        for detail in rec['details']:
            print(f"  • {detail}")
    
    # Save results
    import json
    with open('optimization_results.json', 'w') as f:
        # Convert numpy types to native Python types for JSON
        results_json = {
            'staff_requirements': {
                'peak_staff': int(results['staff_requirements']['peak_staff']),
                'avg_staff': float(results['staff_requirements']['avg_staff']),
                'total_staff_24h': int(results['staff_requirements']['total_staff_24h']),
                'shifts': results['staff_requirements']['shifts']
            },
            'bed_assessment': {
                'max_icu_utilization': float(results['bed_assessment']['max_icu_utilization']),
                'avg_icu_utilization': float(results['bed_assessment']['avg_icu_utilization']),
                'critical_hours_count': len(results['bed_assessment']['critical_hours']),
                'alerts': results['bed_assessment']['alerts']
            },
            'preparedness_plan': results['preparedness_plan']
        }
        json.dump(results_json, f, indent=2)
    
    print("\n\nResults saved to optimization_results.json")
    
    return results

if __name__ == "__main__":
    demo()
