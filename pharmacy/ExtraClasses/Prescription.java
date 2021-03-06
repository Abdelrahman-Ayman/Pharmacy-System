package pharmacy.ExtraClasses;
import pharmacy.ProductClasses.PrescriptionMedication;
import java.util.ArrayList;

public class Prescription {
    private int code;
    private DateTime date;
    private ArrayList <PrescriptionMedication> Medications;

    public Prescription(int code, ArrayList<PrescriptionMedication> Medications) {
        this.code = code;
        this.Medications = Medications;
    }

    public int getCode() {
        return code;
    }

    public void setCode(int code) {
        this.code = code;
    }

    public ArrayList<PrescriptionMedication> getMedications() {
        return Medications;
    }

    public void setMedications(ArrayList<PrescriptionMedication> Medications) {
        this.Medications = Medications;
    }
    
    public void addMedication(PrescriptionMedication pm){
        this.Medications.add(pm);
    }
    public DateTime getDate() {
        return date;
    }

    public void setDate(DateTime date) {
        this.date = date;
    }
    
    public boolean isContainMedication (int Code) {
        for (PrescriptionMedication current : Medications) {
            if (Code == current.getCode()) {
                return true;
            }
        }
        return false;
    }
}
