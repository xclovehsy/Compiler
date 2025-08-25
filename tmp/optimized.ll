; ModuleID = '/home/xucong24/Compiler/tmp/optimized.bc'
source_filename = "/home/xucong24/Compiler/datasets/qsort.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@__const.main.arr = private unnamed_addr constant [6 x i32] [i32 10, i32 7, i32 8, i32 9, i32 1, i32 5], align 16
@.str = private unnamed_addr constant [16 x i8] c"Sorted array: \0A\00", align 1
@.str.1 = private unnamed_addr constant [4 x i8] c"%d \00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@str = private unnamed_addr constant [15 x i8] c"Sorted array: \00", align 1

; Function Attrs: nounwind uwtable
define dso_local void @quickSort(ptr noundef %0, i32 noundef %1, i32 noundef %2) #0 {
  %4 = icmp slt i32 %1, %2
  br i1 %4, label %5, label %9

5:                                                ; preds = %3
  %6 = call i32 @partition(ptr noundef %0, i32 noundef %1, i32 noundef %2)
  %7 = add nsw i32 %6, -1
  call void @quickSort(ptr noundef %0, i32 noundef %1, i32 noundef %7)
  %8 = add nsw i32 %6, 1
  call void @quickSort(ptr noundef %0, i32 noundef %8, i32 noundef %2)
  br label %9

9:                                                ; preds = %5, %3
  ret void
}

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: nounwind uwtable
define dso_local i32 @partition(ptr noundef %0, i32 noundef %1, i32 noundef %2) #0 {
  %4 = sext i32 %2 to i64
  %5 = getelementptr inbounds i32, ptr %0, i64 %4
  %6 = load i32, ptr %5, align 4, !tbaa !5
  %7 = add nsw i32 %1, -1
  br label %8

8:                                                ; preds = %39, %3
  %.0 = phi i32 [ %7, %3 ], [ %.1, %39 ]
  %storemerge = phi i32 [ %1, %3 ], [ %40, %39 ]
  %.not.not = icmp slt i32 %storemerge, %2
  br i1 %.not.not, label %23, label %9

9:                                                ; preds = %8
  %10 = add nsw i32 %.0, 1
  %11 = sext i32 %10 to i64
  %12 = getelementptr inbounds i32, ptr %0, i64 %11
  %13 = load i32, ptr %12, align 4, !tbaa !5
  %14 = sext i32 %2 to i64
  %15 = getelementptr inbounds i32, ptr %0, i64 %14
  %16 = load i32, ptr %15, align 4, !tbaa !5
  %17 = add nsw i32 %.0, 1
  %18 = sext i32 %17 to i64
  %19 = getelementptr inbounds i32, ptr %0, i64 %18
  store i32 %16, ptr %19, align 4, !tbaa !5
  %20 = sext i32 %2 to i64
  %21 = getelementptr inbounds i32, ptr %0, i64 %20
  store i32 %13, ptr %21, align 4, !tbaa !5
  %22 = add nsw i32 %.0, 1
  ret i32 %22

23:                                               ; preds = %8
  %24 = sext i32 %storemerge to i64
  %25 = getelementptr inbounds i32, ptr %0, i64 %24
  %26 = load i32, ptr %25, align 4, !tbaa !5
  %.not = icmp sgt i32 %26, %6
  br i1 %.not, label %39, label %27

27:                                               ; preds = %23
  %28 = add nsw i32 %.0, 1
  %29 = sext i32 %28 to i64
  %30 = getelementptr inbounds i32, ptr %0, i64 %29
  %31 = load i32, ptr %30, align 4, !tbaa !5
  %32 = sext i32 %storemerge to i64
  %33 = getelementptr inbounds i32, ptr %0, i64 %32
  %34 = load i32, ptr %33, align 4, !tbaa !5
  %35 = sext i32 %28 to i64
  %36 = getelementptr inbounds i32, ptr %0, i64 %35
  store i32 %34, ptr %36, align 4, !tbaa !5
  %37 = sext i32 %storemerge to i64
  %38 = getelementptr inbounds i32, ptr %0, i64 %37
  store i32 %31, ptr %38, align 4, !tbaa !5
  br label %39

39:                                               ; preds = %23, %27
  %.1 = phi i32 [ %.0, %23 ], [ %28, %27 ]
  %40 = add nsw i32 %storemerge, 1
  br label %8, !llvm.loop !9
}

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: nounwind uwtable
define dso_local i32 @main() #0 {
  %1 = alloca [6 x i32], align 16
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %1) #5
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 16 dereferenceable(24) %1, ptr noundef nonnull align 16 dereferenceable(24) @__const.main.arr, i64 24, i1 false)
  call void @quickSort(ptr noundef nonnull %1, i32 noundef 0, i32 noundef 5)
  %puts = call i32 @puts(ptr nonnull @str)
  br label %2

2:                                                ; preds = %5, %0
  %storemerge = phi i32 [ 0, %0 ], [ %10, %5 ]
  %3 = icmp slt i32 %storemerge, 6
  br i1 %3, label %5, label %4

4:                                                ; preds = %2
  %putchar = call i32 @putchar(i32 10)
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %1) #5
  ret i32 0

5:                                                ; preds = %2
  %6 = sext i32 %storemerge to i64
  %7 = getelementptr inbounds [6 x i32], ptr %1, i64 0, i64 %6
  %8 = load i32, ptr %7, align 4, !tbaa !5
  %9 = call i32 (ptr, ...) @printf(ptr noundef nonnull @.str.1, i32 noundef %8) #5
  %10 = add nsw i32 %storemerge, 1
  br label %2, !llvm.loop !12
}

; Function Attrs: argmemonly nocallback nofree nounwind willreturn
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #2

declare i32 @printf(ptr noundef, ...) #3

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) #4

; Function Attrs: nofree nounwind
declare noundef i32 @putchar(i32 noundef) #4

attributes #0 = { nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { argmemonly nocallback nofree nosync nounwind willreturn }
attributes #2 = { argmemonly nocallback nofree nounwind willreturn }
attributes #3 = { "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { nofree nounwind }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"Ubuntu clang version 15.0.7"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = distinct !{!9, !10, !11}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!"llvm.loop.unroll.disable"}
!12 = distinct !{!12, !10, !11}
