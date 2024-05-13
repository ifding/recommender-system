"use strict";(self.webpackChunkrecommender_system=self.webpackChunkrecommender_system||[]).push([[194],{160:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>l,contentTitle:()=>a,default:()=>h,frontMatter:()=>r,metadata:()=>o,toc:()=>c});var s=t(4848),i=t(8453);const r={sidebar_position:2},a="Instagram Explore",o={id:"basics/Instagram",title:"Instagram Explore",description:"2023/08/09",source:"@site/docs/basics/Instagram.md",sourceDirName:"basics",slug:"/basics/Instagram",permalink:"/recommender-system/docs/basics/Instagram",draft:!1,unlisted:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/basics/Instagram.md",tags:[],version:"current",sidebarPosition:2,frontMatter:{sidebar_position:2},sidebar:"tutorialSidebar",previous:{title:"Deep Learning Recommendation Models (DLRM)",permalink:"/recommender-system/docs/basics/DLRM"},next:{title:"Markdown Features",permalink:"/recommender-system/docs/basics/markdown-features"}},l={},c=[{value:"Retrieval",id:"retrieval",level:2},{value:"Two Tower NN",id:"two-tower-nn",level:3},{value:"User interactions history",id:"user-interactions-history",level:3},{value:"Ranking",id:"ranking",level:2},{value:"First-stage ranking",id:"first-stage-ranking",level:3},{value:"Second-stage ranking",id:"second-stage-ranking",level:3},{value:"Final reranking",id:"final-reranking",level:2},{value:"Reference",id:"reference",level:2}];function d(e){const n={a:"a",blockquote:"blockquote",code:"code",h1:"h1",h2:"h2",h3:"h3",img:"img",li:"li",ol:"ol",p:"p",strong:"strong",ul:"ul",...(0,i.R)(),...e.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(n.h1,{id:"instagram-explore",children:"Instagram Explore"}),"\n",(0,s.jsxs)(n.blockquote,{children:["\n",(0,s.jsx)(n.p,{children:"2023/08/09"}),"\n"]}),"\n",(0,s.jsxs)(n.blockquote,{children:["\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.a,{href:"https://ai.meta.com/blog/powered-by-ai-instagrams-explore-recommender-system/",children:"Explore"})," is one of the largest recommendation systems on Instagram."]}),"\n"]}),"\n",(0,s.jsx)(n.p,{children:(0,s.jsx)(n.img,{alt:"Explore",src:t(375).A+"",width:"1056",height:"968"})}),"\n",(0,s.jsxs)(n.blockquote,{children:["\n",(0,s.jsx)(n.p,{children:"The stages funnel for Explore on Instagram."}),"\n"]}),"\n",(0,s.jsx)(n.p,{children:"As the system has continued to evolve, we\u2019ve expanded our multi-stage ranking approach with several well-defined stages, each focusing on different objectives and algorithms."}),"\n",(0,s.jsxs)(n.ol,{children:["\n",(0,s.jsx)(n.li,{children:"Retrieval"}),"\n",(0,s.jsx)(n.li,{children:"First-stage ranking"}),"\n",(0,s.jsx)(n.li,{children:"Second-stage ranking"}),"\n",(0,s.jsx)(n.li,{children:"Final reranking"}),"\n"]}),"\n",(0,s.jsx)(n.h2,{id:"retrieval",children:"Retrieval"}),"\n",(0,s.jsx)(n.p,{children:"The basic idea behind retrieval is to get an approximation of what content (candidates) will be ranked high at later stages in the process if all of the content is drawn from a general media distribution."}),"\n",(0,s.jsx)(n.p,{children:"In a world with infinite computational power and no latency requirements we could rank all possible content. But, given real-world requirements and constraints, most large-scale recommender systems employ a multi-stage funnel approach \u2013 starting with thousands of candidates and narrowing down the number of candidates to hundreds as we go down the funnel."}),"\n",(0,s.jsx)(n.p,{children:"The retrieval stage consists of multiple candidates\u2019 retrieval sources (\u201csources\u201d for short). The main purpose of a source is to select hundreds of relevant items from a media pool of billions of items."}),"\n",(0,s.jsx)(n.p,{children:"Candidates\u2019 sources can be based on heuristics (e.g., trending posts) as well as more sophisticated ML approaches. Additionally, retrieval sources can be real-time (capturing most recent interactions) and pre-generated (capturing long-term interests)."}),"\n",(0,s.jsx)(n.p,{children:(0,s.jsx)(n.img,{src:t(8736).A+"",width:"941",height:"701"})}),"\n",(0,s.jsxs)(n.blockquote,{children:["\n",(0,s.jsx)(n.p,{children:"The four types of retrieval sources."}),"\n"]}),"\n",(0,s.jsx)(n.p,{children:"To model media retrieval for different user groups with various interests, we utilize all these mentioned source types together and mix them with tunable weights."}),"\n",(0,s.jsx)(n.p,{children:"Let\u2019s take a closer look at a couple of techniques that can be used in retrieval."}),"\n",(0,s.jsx)(n.h3,{id:"two-tower-nn",children:"Two Tower NN"}),"\n",(0,s.jsx)(n.p,{children:(0,s.jsx)(n.img,{src:t(8950).A+"",width:"858",height:"541"})}),"\n",(0,s.jsxs)(n.ol,{children:["\n",(0,s.jsx)(n.li,{children:"The Two Tower model consists of two separate neural networks \u2013 one for the user and one for the item."}),"\n",(0,s.jsx)(n.li,{children:"Each neural network only consumes features related to their entity and outputs an embedding."}),"\n",(0,s.jsx)(n.li,{children:"The learning objective is to predict engagement events (e.g., someone liking a post) as a similarity measure between user and item embeddings."}),"\n",(0,s.jsx)(n.li,{children:"After training, user embeddings should be close to the embeddings of relevant items for a given user."}),"\n"]}),"\n",(0,s.jsx)(n.p,{children:"Therefore, item embeddings close to the user\u2019s embedding can be used as candidates for ranking."}),"\n",(0,s.jsx)(n.p,{children:(0,s.jsx)(n.img,{src:t(2406).A+"",width:"1155",height:"649"})}),"\n",(0,s.jsxs)(n.blockquote,{children:["\n",(0,s.jsx)(n.p,{children:"How the Two Towers model handles retrieval."}),"\n"]}),"\n",(0,s.jsx)(n.p,{children:"Given that user and item networks (towers) are independent after training, we can use an item tower to generate embeddings for items that can be used as candidates. And we can do this on a daily basis using an offline pipeline."}),"\n",(0,s.jsx)(n.p,{children:"We can also put generated item embeddings into a service that supports online approximate nearest neighbors (ANN) search (e.g., FAISS, HNSW, etc)."}),"\n",(0,s.jsx)(n.p,{children:"During online retrieval we use the user tower to generate user embedding on the fly by fetching the freshest user-side features, and use it to find the most similar items in the ANN service."}),"\n",(0,s.jsx)(n.p,{children:"The main advantage of the Two Tower approach is that user and item embeddings can be cached, making inference for the Two Tower model extremely efficient."}),"\n",(0,s.jsx)(n.h3,{id:"user-interactions-history",children:"User interactions history"}),"\n",(0,s.jsx)(n.p,{children:"We can also use item embeddings directly to retrieve similar items to those from a user\u2019s interactions history."}),"\n",(0,s.jsx)(n.p,{children:"Let\u2019s say that a user liked/saved/shared some items. Given that we have embeddings of those items, we can find a list of similar items to each of them and combine them into a single list."}),"\n",(0,s.jsx)(n.p,{children:"This list will contain items reflective of the user\u2019s previous and current interests."}),"\n",(0,s.jsx)(n.p,{children:(0,s.jsx)(n.img,{src:t(817).A+"",width:"1155",height:"649"})}),"\n",(0,s.jsx)(n.h2,{id:"ranking",children:"Ranking"}),"\n",(0,s.jsx)(n.p,{children:"After candidates are retrieved, the system needs to rank them by value to the user."}),"\n",(0,s.jsx)(n.p,{children:"In Explore, because it\u2019s infeasible to rank all candidates using heavy models, we use two stages:"}),"\n",(0,s.jsxs)(n.ol,{children:["\n",(0,s.jsx)(n.li,{children:"A first-stage ranker (i.e., lightweight model), which is less precise and less computationally intensive and can recall thousands of candidates."}),"\n",(0,s.jsx)(n.li,{children:"A second-stage ranker (i.e., heavy model), which is more precise and compute intensive and operates on the 100 best candidates from the first stage."}),"\n"]}),"\n",(0,s.jsx)(n.h3,{id:"first-stage-ranking",children:"First-stage ranking"}),"\n",(0,s.jsx)(n.p,{children:"In the first-stage ranking our old friend the Two Tower NN comes into play again because of its cacheability property."}),"\n",(0,s.jsx)(n.p,{children:(0,s.jsx)(n.img,{src:t(3080).A+"",width:"826",height:"550"})}),"\n",(0,s.jsxs)(n.blockquote,{children:["\n",(0,s.jsx)(n.p,{children:"Two Tower inference with caching on the both the user and item side."}),"\n"]}),"\n",(0,s.jsx)(n.p,{children:"Even though the model architecture could be similar to retrieval, the learning objective differs quite a bit: We train the first stage ranker to predict the output of the second stage with the label:"}),"\n",(0,s.jsx)(n.p,{children:(0,s.jsx)(n.code,{children:"PSelect = { media in top K results ranked by the second stage}"})}),"\n",(0,s.jsx)(n.p,{children:"We can view this approach as a way of distilling knowledge from a bigger second-stage model to a smaller (more light-weight) first-stage model."}),"\n",(0,s.jsx)(n.h3,{id:"second-stage-ranking",children:"Second-stage ranking"}),"\n",(0,s.jsx)(n.p,{children:"After the first stage we apply the second-stage ranker, which predicts the probability of different engagement events (click, like, etc.) using the multi-task multi label (MTML) neural network model."}),"\n",(0,s.jsx)(n.p,{children:"The MTML model is much heavier than the Two Towers model. But it can also consume the most powerful user-item interaction features."}),"\n",(0,s.jsx)(n.p,{children:"Applying a much heavier MTML model during peak hours could be tricky. That\u2019s why we precompute recommendations for some users during off-peak hours. This helps ensure the availability of our recommendations for every Explore user."}),"\n",(0,s.jsxs)(n.p,{children:["In order to produce a final score that we can use for ordering of ranked items, predicted probabilities for P(click), P(like), P(see less), etc. could be combined with weights W_click, W_like, and W_see_less using a formula that we call ",(0,s.jsx)(n.strong,{children:"value model"})," (VM)."]}),"\n",(0,s.jsx)(n.p,{children:"VM is our approximation of the value that each media brings to a user."}),"\n",(0,s.jsx)(n.p,{children:(0,s.jsx)(n.code,{children:"Expected Value = W_click * P(click) + W_like * P(like) \u2013 W_see_less * P(see less) + etc."})}),"\n",(0,s.jsx)(n.p,{children:"Tuning the weights of the VM allows us to explore different tradeoffs between online engagement metrics."}),"\n",(0,s.jsx)(n.p,{children:"For example, by using higher W_like weight, final ranking will pay more attention to the probability of a user liking a post. Because different people might have different interests in regards to how they interact with recommendations it\u2019s very important that different signals are taken into account. The end goal of tuning weights is to find a good tradeoff that maximizes our goals without hurting other important metrics."}),"\n",(0,s.jsx)(n.h2,{id:"final-reranking",children:"Final reranking"}),"\n",(0,s.jsx)(n.p,{children:"Simply returning results sorted with reference to the final VM score might not be always a good idea. For example, we might want to filter-out/downrank some items based on integrity-related scores (e.g., removing potentially harmful content)."}),"\n",(0,s.jsx)(n.p,{children:"Also, in case we would like to increase the diversity of results, we might shuffle items based on some business rules (e.g., \u201cDo not show items from the same authors in a sequence\u201d)."}),"\n",(0,s.jsx)(n.p,{children:"Applying these sorts of rules allows us to have a much better control over the final recommendations, which helps to achieve better online engagement."}),"\n",(0,s.jsx)(n.h2,{id:"reference",children:"Reference"}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:["Scaling the Instagram Explore recommendations system, ",(0,s.jsx)(n.a,{href:"https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/",children:"https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/"})]}),"\n"]})]})}function h(e={}){const{wrapper:n}={...(0,i.R)(),...e.components};return n?(0,s.jsx)(n,{...e,children:(0,s.jsx)(d,{...e})}):d(e)}},3080:(e,n,t)=>{t.d(n,{A:()=>s});const s=t.p+"assets/images/cacheability-5e935300b54ee2d55a3f71b6b1bf10af.png"},375:(e,n,t)=>{t.d(n,{A:()=>s});const s=t.p+"assets/images/explore-5e79293325e80a948c4c7a4c5196e3b9.png"},8736:(e,n,t)=>{t.d(n,{A:()=>s});const s=t.p+"assets/images/sources-1f85a00cc4118766d60bf2fac8c4878b.png"},8950:(e,n,t)=>{t.d(n,{A:()=>s});const s=t.p+"assets/images/two_tower-0cf5801d710eee1000c9d9c54423fa24.png"},2406:(e,n,t)=>{t.d(n,{A:()=>s});const s=t.p+"assets/images/two_tower_inference-1b68d36a94a6e5df881af99a4ee9157b.png"},817:(e,n,t)=>{t.d(n,{A:()=>s});const s=t.p+"assets/images/user_interaction_history-fc643bd2f8a13355bc73d08939dd7c5e.png"},8453:(e,n,t)=>{t.d(n,{R:()=>a,x:()=>o});var s=t(6540);const i={},r=s.createContext(i);function a(e){const n=s.useContext(r);return s.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function o(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(i):e.components||i:a(e.components),s.createElement(r.Provider,{value:n},e.children)}}}]);