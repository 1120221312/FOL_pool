# FOL_pool
Deep Interaction Timing: FOL-based Complexity Differentiation for Legal Queries

Our core experiments: First, we validated the effectiveness of the rule reasoning pipeline (See Table 3, Line 474); second, We used this pipeline to generate a batch of training data, which was then used for cross-training and fine-tuning a 7b model to acquire rule reasoning abilities (See Table 3, Line 475).

Pipeline：

    Step 1: Data Preparation
①Prepare case generation questions (3 types of granularity);
②Prepare cause/dispute annotations, along with specific law articles under each point, and convert them into law rules. This will be used for in-context learning to expand the pool.

• Query: My husband admitted to incurring a debt of 200,000 during our marriage. He claims this was borrowed to supplement our living expenses, but I believe the money was not used for our family’s expenditures. Will this be considered joint debt?  
• Cause of Action: <map Sub-pool>  Marital property division dispute

    Step 2: Construction of the FOL Pool
Our knowledge FOL pool is a structured collection of 870 civil case sub-pools, each organized hierarchically as follows:
①Sub-pool Layer:
Each sub-pool represents a distinct civil case type, (e.g., Marital Property Division Disputes).
②Controversy Layer:
Within each sub-pool, multiple controversy categories exist (e.g., Common Property Identification).
③Rule Formation Layer:
Each controversy category includes:
    Multiple applicable legal articles (e.g., Civil Code Art. 1062)
    Corresponding FOL rules derived from articles (e.g., ∀p[Property(p)...])
    
    Step 3: Matching Process
Given a query-cause pair, our model identifies the relevant sub-pool. It then iteratively checks whether each disputed issue exists, based on the model's semantic understanding. For each exists, the model selects the most applicable legal rules from one or more related articles.  

• Query: My husband admitted to incurring a debt of 200,000…Will this be considered joint debt?  
• Sub-pool: Marital property division dispute → Controversy: Identification of joint debt → Article 1064 → ∀A [MaritalDebt(A) ⇔ (∃B (JointSignature(B) ∨ PostApproval(A, B)) ∨ (Debt(A) ∧ DailyNeeds(A)))]  

    Step 4: Logical Inference
Using the matched query-rule pair, we perform logical reasoning to assess whether all necessary elements of the rule are present. If any key element is missing, the system requests additional user input to complete the reasoning. If all elements are present and no conflicts exist between rules, a clear and accurate conclusion can be derived.
Our large language model (LLM) uses in-context learning to analyze the query-rule pair in a manner similar to Prolog’s reasoning model, determining whether the conclusion elements can be logically inferred.
• Query fact：
       borrower(husband).
       loan_amount(200,000).
       debt(husband_loan).            
       marriage_status(valid).
• Rules fact:
       marital_debt(Debt) :-
           joint_signature(Debt);
           post_approval(Debt, _);
       (debt(Debt), daily_needs(Debt)).
• check query:
       marital_debt(husband_loan) = FALSE.
This indicates that the model concludes the verification of the conclusion element has failed. The reason for this is the lack of proof or basis to establish "DailyNeeds" as a valid condition, preventing the inference from being properly verified as joint debt.

