# ============================================================================
# COMPLETE FIX FOR NLLB - Both Dataset AND Training Args
# ============================================================================

# ============================================================================
# STEP 1: CELL 4 - Complete Dataset Class (Copy Everything Below)
# ============================================================================

from torch.utils.data import Dataset

class DirectionalTranslationDataset(Dataset):
    def __init__(self, data_path, tokenizer, direction, max_length=128, model_type='mbart'):
        self.df = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.direction = direction
        self.max_length = max_length

        # Language codes
        if model_type == 'mbart':
            self.lang_codes = {'en': 'en_XX', 'ne': 'ne_NP'}
        else:
            self.lang_codes = {'en': 'eng_Latn', 'ne': 'npi_Deva'}

        # Set source and target
        if direction == 'ne_en':
            self.source_col, self.target_col = 'Nepali', 'English'
            self.src_lang = self.lang_codes['ne']
            self.tgt_lang = self.lang_codes['en']
        else:
            self.source_col, self.target_col = 'English', 'Nepali'
            self.src_lang = self.lang_codes['en']
            self.tgt_lang = self.lang_codes['ne']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # ✅ Set language codes for this sample
        if hasattr(self.tokenizer, 'src_lang'):
            self.tokenizer.src_lang = self.src_lang
        if hasattr(self.tokenizer, 'tgt_lang'):
            self.tokenizer.tgt_lang = self.tgt_lang

        source = str(self.df.iloc[idx][self.source_col])
        target = str(self.df.iloc[idx][self.target_col])

        # ✅ Return as lists (NO return_tensors='pt')
        model_inputs = self.tokenizer(
            source,
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        labels = self.tokenizer(
            text_target=target,
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        # ✅ Return as lists (NO .squeeze())
        return {
            'input_ids': model_inputs['input_ids'],
            'attention_mask': model_inputs['attention_mask'],
            'labels': labels['input_ids']
        }

print("✅ Dataset class defined")

# ============================================================================
# STEP 2: CELL 5 - In Seq2SeqTrainingArguments, find this line:
# ============================================================================

# CHANGE FROM:
        remove_unused_columns=(model_type == 'nllb'),

# TO:
        remove_unused_columns=True,

# ============================================================================
# COMPLETE INSTRUCTIONS:
# ============================================================================
# 1. Replace ENTIRE Cell 4 with the code above (Dataset class)
# 2. In Cell 5, change: remove_unused_columns=True
# 3. Re-run Cell 4
# 4. Re-run Cell 5
# 5. Run Cell 9 (NLLB training)
#
# THIS WORKS FOR BOTH MODELS:
# - mBART: Will work with lists + remove_unused_columns=True
# - NLLB: Will work with lists + remove_unused_columns=True
# - Fully reproducible for conference submission
# ============================================================================
