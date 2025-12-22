# ======================================
# Step 1: જરૂરી libraries import કરીએ
# ======================================

# CountVectorizer:
# → Text ને numbers (word count) માં બદલે
from sklearn.feature_extraction.text import CountVectorizer

# LinearRegression:
# → numbers પરથી score શીખે (y = w1*x1 + w2*x2 + ... + b)
from sklearn.linear_model import LinearRegression


# ======================================
# Step 2: Training data (Experience E)
# ======================================

# Input reviews (text)
reviews = [
    "slow boring movie",        # sentiment ≈ very negative
    "not good bad film",        # sentiment ≈ negative
    "average movie",            # sentiment ≈ neutral
    "good movie enjoyed",       # sentiment ≈ positive
    "wow amazing loved it"      # sentiment ≈ very positive
]

# Output scores (actual ratings)
scores = [-2.5, -2.0, 0.0, 1.5, 2.8]


# ======================================
# Step 3: Text → Numbers
# ======================================

# vectorizer = VARIABLE
# જે CountVectorizer class નો OBJECT store કરે છે
vectorizer = CountVectorizer()

# fit_transform = fit + transform
# fit(): reviews માંથી vocabulary બનાવે
# Example vocabulary (approx):
# {
#   'slow':0, 'boring':1, 'movie':2, 'not':3, 'good':4,
#   'bad':5, 'film':6, 'average':7, 'enjoyed':8,
#   'wow':9, 'amazing':10, 'loved':11, 'it':12
# }

X = vectorizer.fit_transform(reviews)

# X (conceptually) looks like this:
# "slow boring movie"   → [1,1,1,0,0,0,0,0,0,0,0,0,0]
# "not good bad film"   → [0,0,0,1,1,1,1,0,0,0,0,0,0]
# "average movie"       → [0,0,1,0,0,0,0,1,0,0,0,0,0]
# "good movie enjoyed"  → [0,0,1,0,1,0,0,0,1,0,0,0,0]
# "wow amazing loved it"→ [0,0,0,0,0,0,0,0,0,1,1,1,1]

# X internally sparse matrix છે (ઘણા 0 હોવાથી memory save થાય)


# ======================================
# Step 4: Model training
# ======================================

# model = LinearRegression object
model = LinearRegression()

# fit(X, scores):
# model શીખે:
# slow    → negative weight (≈ -0.9)
# boring  → negative weight (≈ -1.1)
# bad     → negative weight (≈ -1.0)
# good    → positive weight (≈ +1.0)
# amazing → positive weight (≈ +1.6)
# loved   → positive weight (≈ +1.4)
model.fit(X, scores)

# After training:
# model.coef_      → word weights
# model.intercept_ → bias (e.g. ≈ 0.1)


# ======================================
# Step 5: New review prediction (Task T)
# ======================================

# New unseen review
new_review = ["such","nice"]

# transform ONLY (fit નહીં)
# કારણ કે "such" vocabulary માં નથી
new_X = vectorizer.transform(new_review)

# new_X vector becomes:
# [0,0,0,0,0,0,0,0,0,0,0,0,0]

predicted_score = model.predict(new_X)

# Prediction logic:
# score = Σ(word_count × weight) + bias
# score = 0 + bias ≈ 0

print("Predicted score:", predicted_score[1])
