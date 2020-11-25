Κ23γ: Ανάπτυξη Λογισμικού για Αλγοριθμικά Προβλήματα

2η Προγραμματιστική Εργασία

Ομάδα 16
Αναστάσιος Αντωνόπουλος - 1115201400014
Θεόδωρος Φιλιππίδης - 1115201500170

Github link: https://github.com/FTheodore/Project_2


Εκδόσεις βιβλιοθηκών στις οποίες αναπτύχθηκαν τα μοντέλα:
    - tensorflow: 2.3.0
    - tensorflow-gpu: 2.3.1
    - keras: 2.4.3

Usage:
    - Autoencoder (N1): python autoencoder.py [ –d <dataset> ]

    - Classifier (N2):  python  classification.py [ –d  <training  set>  –dl  <training  labels> -t <testset> -tl <test labels> -model <autoencoder h5> ]


Γενική περιγραφή προγραμμάτων:
    - Autoencoder (N1): Ο autoencoder έχει υλοποιηθεί έτσι ώστε να γίνεται δυναμική κατασκευή της αρχιτεκτονικής και των layers του νευρωνικού δικτύου.
        Για κάθε layer ο χρήστης έχει την επιλογή να τοποθετήσει είτε convolution layer είτε pooling layer. Τα convolution layers πάντα ακολουθούνται από
        batch normalization layer. Υπάρχει όριο στα pooling layers(=2) για να προκύψουν οι επιθυμητές διαστάσεις. Επίσης δεν επιτρέπεται ο χρήστης να
        τοποθετεί συνεχόμενα pooling layers είτε pooling layers χωρίς να έχει προηγηθεί κάποιο convolution layer. Αυτό έχει γίνει ώστε να κατευθυνθεί ο χρήστης
        στην κατασκευή ορθών και λογικών μοντέλων.

        ΣΗΜΕΙΩΣΗ: Επιπλέον υπάρχει δυνατότητα φόρτωσης προκατασκευασμένου και προεκπαιδευμένου μοντέλου. Σε αυτή την περίπτωση ο χρήστης αρκεί να εισάγει
            μόνο τις καινούργιες υπερπαραμέτρους εκπαίδευσης. Επιπρόσθετα για λόγους έρευνας, εκτός από τις ζητούμενες ενέργειες, κατα την ολοκλήρωση της εκπαίδευσης
            υπάρχει η δυνατότητα πέρα από την αποθήκευση ολόκληρου του μοντέλου και η αποθήκευση των training και validation losses που υπολογίστηκαν σε κάθε
            epoch της εκπαίδευσης.

    - Classifier (N2): Ο classifier εκτός από τα layers που θα κληροδοτήσει από το encoding part του φορτωμένου autoencoder, μπορεί να περιέχει και dropout layers μετά
        τα convolution layers για την αποφυγή του overfitting. Dropout layer μπορέι να τοποθετηθεί και μετά το fully connected layer του classifier. Δίνεται η δυνατότητα
        επιλογής στον χρήστη αν θέλει να τοποθετήσει dropout layers με οποιονδήποτε συνδυασμό των παραπάνω τρόπων καθώς και οι πιθανότητες που έχει κάθε νευρώνας να γίνει
        dropped out.
        ΠΡΟΤΕΙΝΕΤΑΙ η χρήση και των δύο τεχνικών προκειμένου να αποφευχθεί κατά βέλτιστο τρόπο το overfitting.

        Όπως ζητείται από την εκφώνηση, η εκπαίδευση του classifier γίνεται σε δύο στάδια. Αρχικά εκπαιδεύονται μόνο τα weights του fully connected layer.
        Στο δεύτερο στάδιο εκπαιδεύεται το μοντέλο συνολικά.
	
	ΣΗΜΕΙΩΣΗ: Και στον classifier υπάρχει δυνατότητα φόρτωσης προκατασκευασμένου και προεκπαιδευμένου μοντέλου. Περαιτέρω training δεν πραγματοποιείται. Επιπρόσθετα για λόγους έρευνας, εκτός από τις ζητούμενες ενέργειες, κατα την ολοκλήρωση της εκπαίδευσης
            υπάρχει η δυνατότητα πέρα από την αποθήκευση ολόκληρου του μοντέλου και η αποθήκευση των training και validation losses που υπολογίστηκαν σε κάθε
            epoch της εκπαίδευσης.

Ο πηγαίος κώδικας έχει οργανωθεί με τον παρακάτω τρόπο:
    ./
    ├── autoencoder.py          // Υλοποίηση Autoencoder (N1)
    |
    ├── classification.py       // Υλοποίηση Classifier (N2)
    |
    ├── classification_utils.py // Χρήσιμες methods για τον classifier
    |
    ├── input.py                // Γενικές input methods
    |
    ├── utils.py                // Γενικά utilities
    |
    ├── Research_N1.ipynb       // Jupyter Notebook, σύγκριση διάφορων autoencoding μοντέλων
    |
    ├── Research_N2.ipynb       // Jupyter Notebook, σύγκριση διάφορων classificication μοντέλων
    |
    ├── autoencoder_models      // Φάκελος που περιέχει pretrained autoencoding models
    |   ├── best.h5
    |   ├── best_loss.npy
    |   ├── big_batch.h5
    |   ├── big_batch_loss.npy
    |   ├── complex.h5
    |   └── complex_loss.npy
    |
    └──classifier_models/       // Φάκελος που περιέχει pretrained classification models
	├── fc128.h5
	├── fc128.json
	├── fc256.h5
	├── fc256.json
	├── fc32.h5
	├── fc32.json
	├── fc64.h5
	└── fc64.json

